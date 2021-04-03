import yaml
from datetime import datetime
with open('items.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)

with open('data.yaml') as f:
    docs = yaml.load_all(f, Loader=yaml.FullLoader)
    for doc in docs:
        for k, v in doc.items():
            print(k, '->', v)

users = [{'name': 'John Doe', 'occupation': 'gardener'},
         {'name': 'Lucy Black', 'occupation': 'horse rider'}]

print(yaml.dump(users))

with open('users.yaml', 'w') as f:
    data = yaml.dump(users, f)

now_object = datetime.now()

print('now {}'.format(datetime.now()))

print(now_object.year, now_object.month, now_object.day, now_object.hour, now_object.minute, now_object.second)


