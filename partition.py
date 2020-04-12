import csv
import os
import shutil

data = {}
header = []
id_col = master_cat = sub_cat = article_type = None

def renameFolder(name):
  name = name.strip()
  name = name.title()
  name = name.replace(' ','')
  return name

with open('styles.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  isFirst = True
  for row in csv_reader:
    if isFirst:
      header.extend(row)
      isFirst = False
      id_col = header.index('id')
      master_cat = header.index('masterCategory')
      sub_cat = header.index('subCategory')
      article_type = header.index('articleType')
    else:
      iD = row[id_col]
      master = row[master_cat]
      sub = row[sub_cat]
      article = row[article_type]
      if not data.get(master,None):
        data[master] = {}
        data[master][sub] = {}
        data[master][sub][article] = {}
        data[master][sub][article] = [iD]
      elif not data[master].get(sub,None):
        data[master][sub] = {}
        data[master][sub][article] = {}
        data[master][sub][article] = [iD]
      elif not data[master][sub].get(article, None):
        data[master][sub][article] = {}
        data[master][sub][article] = [iD]
      else:
        data[master][sub][article].append(iD)

def getTrainTestPartition(dataArr):
  length = len(dataArr)
  trainLength = int(len(dataArr) * 0.7)
  return dataArr[:trainLength], dataArr[trainLength:]


path = os.getcwd()
# try:
# os.mkdir("{}/{}".format(path, "sid"))
# print(path)
os.mkdir("{}/{}".format(path, "dataset_images"))

os.mkdir("{}/{}".format(path, "dataset_styles"))

for master in data.keys():
  for sub in data[master].keys():
    for article in data[master][sub].keys():
      new_article = renameFolder(master + "_" + sub + "_" + article)
      os.mkdir("{}/{}/{}".format(path, "dataset_images", new_article))
      os.mkdir("{}/{}/{}".format(path, "dataset_styles", new_article))
      for iD in data[master][sub][article]:
        try:
          shutil.copyfile("{}/images/{}.jpg".format(path,iD), "{}/{}/{}/{}.jpg".format(path, "dataset_images", new_article, iD))
        except:
          continue
        shutil.copyfile("{}/styles/{}.json".format(path,iD), "{}/{}/{}/{}.json".format(path, "dataset_styles", new_article, iD))
print("Done")
