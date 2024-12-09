from http.server import HTTPServer
from os import getenv
from src.models.model_config import ModelEnum
from tinygrad.device import Device
from tinygrad.runtime.ops_mcloud import MCloudHandler
import multiprocessing


def cloud_server(port:int):
  multiprocessing.current_process().name = "MainProcess"
  MCloudHandler.device = Device.DEFAULT
  model = ModelEnum.QWEN_0_5B.value
  weights = model.get_model_weights_on_disk()
  models = {model.hub_name: weights}
  MCloudHandler.add_memory(models)
  print(f"start cloud server on {port} with device {MCloudHandler.device}")
  server = HTTPServer(('', port),MCloudHandler)
  server.serve_forever()

if __name__ == "__main__": cloud_server(int(getenv("PORT", 6667)))
