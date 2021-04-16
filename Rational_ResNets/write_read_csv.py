import csv
import argparser
from datetime import datetime

args = argparser.get_args()


def make_csv(model, epoch, train_acc: list, val_acc: list, test_acc: list):
    time_stamp = datetime.now()
    PATH = 'CSV/{}'.format(model) + '/{}'.format(time_stamp) + '.csv'
    with open(PATH, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(train_acc)):
            writer.writerow({'Epoch': epoch[i], 'Train Accuracy': train_acc[i].cpu().numpy(), 'Validation Accuracy': val_acc[i].cpu().numpy(), 'Test Accuracy': test_acc[i].cpu().numpy()})

    return PATH



