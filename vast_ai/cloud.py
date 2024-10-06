import json
import os
import re
from vastai import VastAI


class VastAIInstance:

  def __init__(self):
    self.vast_sdk = VastAI(api_key=os.getenv('VAST_API_KEY'))

  def parse_result(self, output):
    lines = output.strip().split('\n')

    # Extract header and rows
    header = re.split(r'\s{2,}', lines[0])
    rows = [re.split(r'\s{2,}', line) for line in lines[1:]]

    # Convert rows into list of dictionaries
    return [dict(zip(header, row)) for row in rows]

  def create_instance(self):
    # https://en.wikipedia.org/wiki/GeForce_40_series
    query = "dph<0.161 rentable=True num_gpus=1 gpu_ram>10 gpu_ram<15 disk_space>40 dlperf>30 gpu_name in [RTX_4070_Ti,RTX_4080,RTX_4070] cuda_vers>=12.1"
    output = self.vast_sdk.search_offers(type='bid', query=query, storage=40)
    # decode
    print(output)
    data_dicts = self.parse_result(output)

    self.vast_sdk.create_instance(ID=int(data_dicts[0]['ID']), disk=40, image='nvidia/cuda:12.1.1-devel-ubuntu22.04')

  def destroy(self):
    instances = self.vast_sdk.show_instances()
    data_dicts = self.parse_result(instances)
    print(data_dicts)
    # get all ID
    # Shutdown the instance
    self.vast_sdk.destroy_instances(ids=[int(data_dict['ID']) for data_dict in data_dicts])


client = VastAIInstance()
client.destroy()

# vast_sdk.launch_instance(gpu_name='RTX_4070', num_gpus='1', image='pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel')
