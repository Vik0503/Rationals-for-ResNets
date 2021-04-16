import csv
import argparser

args = argparser.get_args()


def make_csv():
    with open('CSV/names.csv', 'w', newline='') as csvfile:
        fieldnames = ['batch_size', 'last_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'batch_size': '{}'.format(args.batch_size), 'last_name': 'Beans'})


make_csv()


import yaml

dict_file = [{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}]

with open('../LTH_for_Rational_ResNets/YAML/store_file.yaml', 'w') as file:
    documents = yaml.dump(dict_file, file)

