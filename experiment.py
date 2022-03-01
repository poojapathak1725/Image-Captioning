################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from PIL import Image

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        
        if os.path.exists(self.__experiment_dir):
            self.__best_model = torch.load(self.__experiment_dir+'/best_model.pt')
        else:
            self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean")
        self.__optimizer = optim.Adam(self.__model.parameters(), lr = 5e-4)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()
        
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print("------------------------Epoch:" + str(epoch) + "------------------------")
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        device = self.__device
        for i, (images, captions, _) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            images = images.to(device)
            captions = captions.to(device)
            outputs = self.__model(images,captions)
            loss = self.__criterion(outputs.reshape(-1,outputs.shape[2]), captions.reshape(-1))
            training_loss += loss.item()
            print("Loss:",loss.item())
            
            loss.backward()
            self.__optimizer.step()
            
        training_loss /= (i+1)
        print("Average Training Loss:", training_loss)
        return training_loss

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        device = self.__device
        
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to(device)
                captions = captions.to(device)
                outputs = self.__model(images,captions)
                loss = self.__criterion(outputs.reshape(-1,outputs.shape[2]), captions.reshape(-1))
                print("Val Loss:",loss.item())
                val_loss += loss.item()
            val_loss /= (i+1)
            print("Average Val Loss:", val_loss)

        # update best model
        if not self.__val_losses or val_loss < min(self.__val_losses):
            self.__best_model = self.__model
            best_model_path = os.path.join(self.__experiment_dir, "best_model.pt")
            torch.save(self.__best_model, best_model_path)
          
        # turn off evaluation mode
        self.__model.train()
            
        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self, mode="stochastic", temperature=0.1):
        self.__model.eval()
        test_loss = 0
        bl1 = 0
        bl4 = 0
        tot_capts = 0

        device = self.__device

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                print("Testing Batch " + str(i))

                # just to test caption generation
                images = images.to(device)
                captions = captions.to(device)
                
                # output loss
                outputs = self.__best_model(images,captions)
                loss = self.__criterion(outputs.reshape(-1,outputs.shape[2]), captions.reshape(-1))
                test_loss += loss.item()

                pred_captions = self.__best_model.generate_captions(images,mode,temperature)
                
                # for each image get reference captions
                for k, img_id in enumerate(img_ids):
                    ann_ids = [self.__coco_test.imgToAnns[img_id][j]['id'] for j in
                           range(0, len(self.__coco_test.imgToAnns[img_id]))]
                    
                    # convert reference captions to list(list(str))
                    punc = '''!()-[]{};:'"\,./?@#$%^&*_~'''
                    ref_captions = []
                    for ann_id in ann_ids:
                        c = self.__coco_test.anns[ann_id]["caption"].lower()
                        for p in punc:
                            c = c.replace(p,"")
                        ref_captions.append(c.split())
                    
                    # convert predicted caption for respective img to list(str)
                    pred_caption = [self.__vocab.idx2word[word.item()].lower() for word in pred_captions[k]]
                    
                    for pi in range(len(pred_caption)):
                        for p in punc:
                            pred_caption[pi] = pred_caption[pi].replace(p,"")
         
                    # filter tokens
                    for remove_word in ['<pad>', '<start>', '<end>', '<unk>', '']:
                        if remove_word in pred_caption:
                            pred_caption = list(filter((remove_word).__ne__, pred_caption))
                    
                    # calculate bleu1 and bleu4 scores
                    bl1 += bleu1(ref_captions, pred_caption)
                    bl4 += bleu4(ref_captions, pred_caption)
                    
                    # saving 1st image and caption of 1st 10 batches 
                    if k == 0 and i < 10:
                        # get caption as string
                        ik_caption = "".join(p + " " for p in pred_caption)[:-1]
                        ik_ref_caption = []
                        for ref_caption in ref_captions:
                            ik_ref_caption.append("".join(r + " " for r in ref_caption)[:-1] + "\n")
                        
                        # process image to save
                        image = images[0].permute(1,2,0).cpu().detach().numpy()
                        image = image * np.array((0.229,0.224,0.225)) + np.array((0.485,0.456,0.406))
                        image *= 255.0
                        img = Image.fromarray(image.astype(np.uint8))
                        
                        # create files for saving the image and captions
                        img_name = str(self.__current_epoch) + "_" + str(i) + "_test.png"
                        f_name = str(self.__current_epoch) + "_" + str(i) + "_test.txt"
                        ref_name = str(self.__current_epoch) + "_" + str(i) + "_ref_test.txt"
                        img_dir = os.path.join(self.__experiment_dir, "test_captions")
                        img_path = os.path.join(img_dir, img_name)
                        f_path = os.path.join(img_dir, f_name)
                        ref_path = os.path.join(img_dir, ref_name)
                        
                        if not os.path.exists(img_dir):
                            os.mkdir(img_dir)
                        img.save(img_path)
                        
                        with open(f_path,"w") as f:
                            f.write(ik_caption)
                        f.close()

                        with open(ref_path, "w") as f:
                            f.writelines(ik_ref_caption)
                        f.close()
                tot_capts += k
        
            test_loss /= (i+1)
            bl1 /= tot_capts
            bl4 /= tot_capts

        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                               bl1,
                                                                               bl4)
        self.__log(result_str)

        return test_loss, bl1, bl4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
