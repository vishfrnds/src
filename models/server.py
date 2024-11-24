import multiprocessing
from http.server import HTTPServer
from os import getenv

from tinygrad.device import Device
from tinygrad.runtime.ops_cloud import CloudHandler, cloud_server


class CloudHandlerWithMemory(CloudHandler):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.memory = {}

def cloud_server(port:int):
  multiprocessing.current_process().name = "MainProcess"
  CloudHandler.device = getenv("CLOUDDEV", "METAL") if Device.DEFAULT == "CLOUD" else Device.DEFAULT
  print(f"start cloud server on {port} with device {CloudHandler.device}")
  server = HTTPServer(('', port), CloudHandlerWithMemory)
  server.serve_forever()

if __name__ == "__main__": cloud_server(int(getenv("PORT", 6667)))
