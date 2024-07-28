# pip install git+https://github.com/jarvislabsai/jlclient.git
import os

from jlclient import jarvisclient
from jlclient.jarvisclient import *
jarvisclient.token = os.environ['JARVIS_API_KEY']

# Launch GPU Instance
gpu_type = 'RTX6000Ada' # A100, A5000, A6000, RTX6000Ada, RTX5000
num_gpus = 4
storage = 500 # 20-2000
template = 'axolotl' # or 'vllm'

instance: Instance = Instance.create('GPU',
                            gpu_type='RTX6000Ada',
                            num_gpus=num_gpus,
                            storage=storage,
                            template = template,
                            name='gpu instance')

'''
instance attributes:
    gpu_type
    num_gpus
    num_cpus
    storage
    name
    machine_id
    script_id
    is_reserved
    duration
    script_args
    http_ports
    template
    url
    endpoints
    ssh_str
    status
'''

instance.pause()
instance.resume()

#Modifying the parameters like scaling GPUs, switching GPU type, expanding storage, etc.
instance.resume(num_gpus=1,
                gpu_type='RTX5000',
                storage=100)

# kill it
instance.destroy()

>>> instance.ssh_str
'ssh -o StrictHostKeyChecking=no -p 21614 root@sshb.jarvislabs.ai'

>>> instance.script_id
''
>>> instance.script_args
''
>>> instance.url
[URL]

>>> instance.status
'Running'
