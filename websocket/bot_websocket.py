"""
6.2.2025
considering different requiremets as follows, decided to first start with websocket
1) flexibility for gan projection, where purturbe value independent of direction can be fullfilled which is so challenging through http 
2) In gan integration we want to to relize each perturbe among various bot config accoriding to gan training batch size. 
So for each generated pertube if it is valide we generate various items to make flows and fullfil pertube value 
3) we come up with factory design pattern to develop it. (the other option was builder which is suit for more complex objects)

see, project_note to get updated 
"""
import asyncio
import websockets
from typing import List
import psutil, socket, yaml, random

with open("websocket/config.yaml") as stream:
    try:
        cnf = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list
rce_rnd_action_list= []
discovery_action_list= random.choices(list(cnf["rce"].keys()), k=random.randint(1,len(list(cnf["rce"].keys()))))
for action in discovery_action_list:
    rce_rnd_action_list.append(random.choices(cnf["rce"][action], k=random.randint(1,len(cnf["rce"][action]))))
flattened_rce_list = flatten_list(rce_rnd_action_list)
def add_iat_to_action_list(action_lst:list, rnd_min, rnd_max):
    final_dic= {}
    for item in action_lst:
        final_dic[item] = random.randint(rnd_min, rnd_max)
    return final_dic
#------
exfil_rnd_file_list =[]
file_rnd_list = random.choices(list(cnf["exfil"].keys()), k=random.randint(1,len(list(cnf["exfil"].keys()))))
for item in file_rnd_list:
    exfil_rnd_file_list.append(random.choices(cnf["exfil"][item], k=random.randint(1,len(cnf["exfil"][item]))))
flattened_exfil_list = flatten_list(exfil_rnd_file_list)
#file_rnd_list = random.choices(cnf["exfil"], k=random.randint(1, len(cnf["exfil"])))

def create_rnd_action_list (rce_list, exfil_list):
    rce = {}
    exfil = {}
    for i, item in enumerate(rce_list):
        rce[f"rce_{i}"] = item
    if len(exfil_list) > 0:
        for j, item in enumerate(exfil_list):
            exfil[f"exfil_{j}"]= item
    dict_ = (rce | exfil)
    combined = list(dict_.items())
    random.shuffle(combined)
    shuffled_dic = dict (combined)
    return shuffled_dic
#print(flattened_exfil_list)
#print(flattened_rce_list)
print(create_rnd_action_list(flattened_rce_list, flattened_exfil_list))



"""
def get_ip_address(interface='ens19'):
    # Get all network interfaces (and their IP addresses)
    addrs = psutil.net_if_addrs()
    
    # Check if the interface exists
    if interface in addrs:
        # Get the IP address for the specified interface
        for addr in addrs[interface]:
            if addr.family == socket.AF_INET:  # Only check IPv4 address
                return addr.address
    return None

async def connect_to_server():
    uri = "ws://localhost:5000"
    
    # Connect to the WebSocket server
    async with websockets.connect(uri) as websocket:
        print(f"Connected to server: {uri}")
        
       
        client_id = get_ip_address() # send client ip as id
        await websocket.send(client_id)
    
        
        # Receive the server's response
        response = await websocket.recv()
        print(f"Received response: {response}")

# Run the client
if __name__ == "__main__":
    asyncio.run(connect_to_server())
    #pass





class Communication:
    def __init__(self, src2dst_ps:int, src2dst_packets:int, src2dst_bytes:int, dst2src_ps:int, dst2src_packets:int, dst2src_bytes:int, duration: float = None, IAT: float= None):
        self.src2dst_ps = src2dst_ps          
        self.src2dst_packets = src2dst_packets    
        self.src2dst_bytes = src2dst_bytes         
        self.dst2src_ps = dst2src_ps    
        self.dst2src_packets = dst2src_packets    
        self.dst2src_bytes = dst2src_bytes   
        self.duration = duration      # duration of the communication
        self.IAT = IAT                # Inter-arrival time 

    def __repr__(self):
        return f"Communication(src2dst_ps={self.src2dst_ps}, src2dst_packets={self.src2dst_packets}, src2dst_bytes={self.src2dst_bytes},dst2src_ps={self.dst2src_ps},dst2src_packets={self.dst2src_packets},dst2src_bytes={self.dst2src_bytes}, duration={self.duration}, IAT={self.IAT})"

# Specific action types

class RCE(Communication):
    def __init__(self, src2dst_ps:int, src2dst_packets:int, src2dst_bytes:int, dst2src_ps:int, dst2src_packets:int, dst2src_bytes:int, duration: float, IAT: float, commands: List[str]):
        super().__init__(src2dst_ps, src2dst_packets, src2dst_bytes, dst2src_ps,dst2src_packets,dst2src_bytes, duration, IAT)
        self.commands = commands  # List of commands to be executed
    def __repr__(self):
        return f"RCE({super().__repr__()}, commands={self.commands})"

class Upload(Communication):
    def __init__(self, src2dst_ps: int, src2dst_packets: int, src2dst_bytes: int, dst2src_ps: int,dst2src_packets: int,dst2src_bytes:int, duration: float, IAT: float, files: List[str]):
        super().__init__(src2dst_ps, src2dst_packets, src2dst_bytes, dst2src_ps,dst2src_packets,dst2src_bytes, duration, IAT)
        self.files = files  # List of files to upload
    def __repr__(self):
        return f"Upload({super().__repr__()}, files={self.files})"


class BotGenerator:
    def createBot(self, action_type, src2dst_ps, src2dst_packets, src2dst_bytes, dst2src_ps,dst2src_packets,dst2src_bytes, duration, IAT):
        if action_type in ["rce", "RCE"]:
            return RCE(src2dst_ps, src2dst_packets, src2dst_bytes, dst2src_ps,dst2src_packets,dst2src_bytes, duration, IAT, commands=["ls", "pwd"])
        elif action_type in ["upload", "UPLOAD"]:
            return Upload(src2dst_ps, src2dst_packets, src2dst_bytes, dst2src_ps,dst2src_packets,dst2src_bytes, duration, IAT, files=["file1.txt", "file2.txt"])
        else:
            raise ValueError("Unknown action type")

# Usage:
bot = BotGenerator()
bot_obj = bot.createBot("upload", 10, 1000, 5, 500, 120, 2,5,7)
print(bot_obj)
"""