from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
import csv
import sys
import os

print('start')

###need 2 files: csv containing materials project IDs and atom_init.json (generic)

###get API key from materials project login dashboard online
API_KEY='f8lrZkNzIhxvuK21'
mpr = MPRester(API_KEY)

###open file containing IDs
f=open('data_new.csv')
csvfile = csv.reader(f, delimiter=',')
materials_ids = list(csvfile)
f.close()

print(len(materials_ids))

if not os.path.exists('new_data'):
  os.mkdir('new_data')

retries=5
count=0
write_prop=open('bulk_data/targts.csv', 'w')
out=[]
for material_id in materials_ids:

  #print(material_id[0])
  try_count=0
  success=0
  while try_count<retries and success==0:
    try:
      out_temp=mpr.query(criteria={"task_id": material_id[0]}, properties=['final_structure', 'pretty_formula'], max_tries_per_chunk=10, chunk_size=0)
    except Exception as e:
      print(e)
    try_count=try_count+1

  if len(out_temp) != 0 :
    out.append(out_temp[0])
    write_prop.write(str(count)+','+str(out[count]['pretty_formula']) + '\n')
    write_cif=CifWriter(out_temp[0]['final_structure'])
    write_cif.write_file('new_data/'+str(out[count]['pretty_formula'])+'.cif')
    
    count=count+1
    if count % 50 ==0:
      print(count)
      
write_prop.close()
