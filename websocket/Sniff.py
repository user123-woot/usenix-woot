from scapy.all import sniff, TCP, UDP, IP
import threading
import json
import time

CONFIG_FILE = "config.json"
def get_flow_key(src_ip, src_port, dst_ip, dst_port, protocol, vlan_id, tunnel_id):
    """Create a consistent, direction-agnostic flow key (7-tuple)."""
    if (src_ip, src_port) < (dst_ip, dst_port):
        return (src_ip, src_port, dst_ip, dst_port, protocol, vlan_id, tunnel_id)
    else:
        return (dst_ip, dst_port, src_ip, src_port, protocol, vlan_id, tunnel_id)
class Sniffer:
    def __init__(self, config_file=CONFIG_FILE):
        """Initialize sniffer with dynamic configuration."""
        self.config = { "host":"127.0.0.1", "protocol": "tcp", "port": 5000, "interface": "lo", "update_interval": 5 } #self.load_config(config_file)
        self.running = False
        self.thread = None
        self.stats = {}  # Track traffic statistics per IP

    def load_config(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print("Configuration file not found. Using default settings.")
            return {"protocol": "tcp", "port": 5000, "interface": None, "update_interval": 5}
    def packet_callback(self, packet):
        """Accurately count bidirectional packets for a 7-tuple flow with VLAN and tunnel support."""
        if not self.running:
            return

        # Ensure packet has both IP and TCP layers
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return  # Skip non-TCP/IP packets

        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        try:
            # Use Scapy methods to get flow key
            flow_key = get_flow_key(
                ip_layer.src, tcp_layer.sport,
                ip_layer.dst, tcp_layer.dport,
                6,  # TCP protocol number
                getattr(packet, "vlan_id", 0),  # Default VLAN ID to 0 if not present
                getattr(packet, "tunnel_id", 0)  # Default Tunnel ID to 0 if not present
            )

            packet_size = len(packet)

            # Initialize stats if the flow is new
            if flow_key not in self.stats:
                self.stats[flow_key] = {
                    "src2dst_packets": 0, "src2dst_bytes": 0, "src2dst_max_ps": 0,
                    "dst2src_packets": 0, "dst2src_bytes": 0, "dst2src_max_ps": 0
                }

            # Ensure bidirectional tracking: (src_ip, sport) could be in either order in flow_key
            if (ip_layer.src, tcp_layer.sport) < (ip_layer.dst, tcp_layer.dport):
                self.stats[flow_key]["src2dst_packets"] += 1
                self.stats[flow_key]["src2dst_bytes"] += packet_size
                self.stats[flow_key]["src2dst_max_ps"] = max(self.stats[flow_key]["src2dst_max_ps"], packet_size)
            else:
                self.stats[flow_key]["dst2src_packets"] += 1
                self.stats[flow_key]["dst2src_bytes"] += packet_size
                self.stats[flow_key]["dst2src_max_ps"] = max(self.stats[flow_key]["dst2src_max_ps"], packet_size)

        except AttributeError as e:
            print(f"Packet processing error: {e}")
        
       
        """
    def packet_callback(self, packet):
    
        if not self.running:
            return

        if packet.haslayer(TCP) and packet.haslayer(IP):  # Capture only TCP+IP traffic
            ip_layer = packet[IP]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            packet_size = len(packet)

            # Ensure each IP has a dictionary initialized
            if src_ip not in self.stats:
                self.stats[src_ip] = {"src2dst_packets": 0, "src2dst_bytes": 0, "src2dst_max_ps": 0,
                                    "dst2src_packets": 0, "dst2src_bytes": 0, "dst2src_max_ps": 0}
            if dst_ip not in self.stats:
                self.stats[dst_ip] = {"src2dst_packets": 0, "src2dst_bytes": 0, "src2dst_max_ps": 0,
                                    "dst2src_packets": 0, "dst2src_bytes": 0, "dst2src_max_ps": 0}

            # Update statistics for source (outgoing traffic)
            self.stats[src_ip]["src2dst_packets"] += 1
            self.stats[src_ip]["src2dst_bytes"] += packet_size
            self.stats[src_ip]["src2dst_max_ps"] = max(self.stats[src_ip]["src2dst_max_ps"], packet_size)

            # Update statistics for destination (incoming traffic)
            self.stats[dst_ip]["dst2src_packets"] += 1
            self.stats[dst_ip]["dst2src_bytes"] += packet_size
            self.stats[dst_ip]["dst2src_max_ps"] = max(self.stats[dst_ip]["dst2src_max_ps"], packet_size)
        """
    def start(self):
        """Start packet sniffing in a separate thread."""
        if self.running:
            print("Sniffer is already running.")
            return

        self.running = True
        self.thread = threading.Thread(target=self.run_sniffing, daemon=True)
        self.thread.start()

        # Start the statistics printer
        threading.Thread(target=self.print_statistics, daemon=True).start()
        print(f"Packet sniffer started on interface '{self.config['interface']}' monitoring {self.config['protocol'].upper()} port {self.config['port']}")

    def run_sniffing(self):
        """Run the sniffer process."""
        filter_rule = f"{self.config['protocol']} and host {self.config['host']} and port {self.config['port']}"
        print(f"Sniffing started with filter: {filter_rule} on {self.config['interface']}")
        sniff(filter=filter_rule, prn=self.packet_callback, store=0, iface=self.config["interface"]) #Specifies a callback function (self.packet_callback) to be executed on each captured packet.

    def print_statistics(self):
        """Regularly print live statistics at the configured interval."""
        update_interval = self.config["update_interval"]
        while self.running:
            time.sleep(update_interval)
            self.report(live=True)

    def stop(self):
        """Stop packet sniffing and print final statistics."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("\nPacket sniffer stopped. Final statistics:")
        self.report()

    def report(self, live=False):
        """Print captured traffic statistics."""
        if not self.stats:
            print("[No packets captured yet.]")
            return

        print("\n--- Live Traffic Statistics ---" if live else "\n--- Final Traffic Report ---")
        for ip, stats in self.stats.items():
            print(f"IP: {ip}")
            print(f"  ↳ Sent: {stats.get('src2dst_packets', 0)} packets, {stats.get('src2dst_bytes', 0)} bytes (Max PS: {stats.get('src2dst_max_ps', 0)})")
            print(f"  ↳ Received: {stats.get('dst2src_packets', 0)} packets, {stats.get('dst2src_bytes', 0)} bytes (Max PS: {stats.get('dst2src_max_ps', 0)})")

if __name__ == "__main__":
    sniffer = Sniffer()  # Create sniffer instance
    sniffer.start()  # Start sniffing

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        sniffer.stop()  # Stop sniffing and show final statistics


"""
from scapy.all import sniff, TCP, IP
import threading
import yaml

CONFIG_FILE = "websocke/config.yaml"

class Sniffer:
    def __init__(self, server):
        self.server = server
        self.config = self.load_config()
        self.running = False
        self.thread = None

    def load_config(self):
     
        with open(CONFIG_FILE) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    def packet_callback(self, packet):
        
        if not self.running:
            return

        if packet.haslayer(TCP) and packet.haslayer(IP):
            ip_layer = packet[IP]
            tcp_layer = packet[TCP]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            packet_size = len(packet)

            # Check if client exists in tracking
            if src_ip in self.server.clients:
                self.server.clients[src_ip]["src2dst_packets"] += 1
                self.server.clients[src_ip]["src2dst_bytes"] += packet_size
                self.server.clients[src_ip]["src2dst_max_ps"] = max(
                    self.server.clients[src_ip]["src2dst_max_ps"], packet_size
                )

            if dst_ip in self.server.clients:
                self.server.clients[dst_ip]["dst2src_packets"] += 1
                self.server.clients[dst_ip]["dst2src_bytes"] += packet_size
                self.server.clients[dst_ip]["dst2src_max_ps"] = max(
                    self.server.clients[dst_ip]["dst2src_max_ps"], packet_size
                )

    def start(self):
        
        if self.running:
            print("Sniffer is already running.")
            return

        self.running = True
        self.thread = threading.Thread(target=self.run_sniffing, daemon=True)
        self.thread.start()
        print("Packet sniffer started...")

    def run_sniffing(self):
       
        sniff(filter="tcp port 5000", prn=self.packet_callback, store=0)

    def stop(self):
      
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("Packet sniffer stopped.")
sn = Sniffer
"""