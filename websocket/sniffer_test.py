"""
from scapy.all import rdpcap, TCP, IP
from collections import defaultdict

# Function to analyze TCP flow
def analyze_tcp_flows(pcap_file):
    # Read pcap file
    packets = rdpcap(pcap_file)

    # Create a dictionary to store flow statistics
    flows = defaultdict(lambda: {'src2dst': 0, 'dst2src': 0})
    
    # Process packets
    for packet in packets:
        if packet.haslayer(TCP):
            # Extract the 4-tuple (src_ip, src_port, dst_ip, dst_port)
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            
            # Define the flow direction: src->dst or dst->src
            if (src_ip, src_port, dst_ip, dst_port) in flows: 
                flows[(src_ip, src_port, dst_ip, dst_port)]['src2dst'] += 1
            elif (dst_ip, dst_port, src_ip, src_port) in flows:
                flows[(dst_ip, dst_port, src_ip, src_port)]['dst2src'] += 1
            else:
                flows[(src_ip, src_port, dst_ip, dst_port)] = {'src2dst': 1, 'dst2src': 0}
    
    # Print the flow statistics
    for flow, counts in flows.items():
        print(f"Flow {flow}:")
        print(f"  Packets from src to dst: {counts['src2dst']}")
        print(f"  Packets from dst to src: {counts['dst2src']}")
        print()

# Call the function with a sample pcap file
analyze_tcp_flows('/home/mehrdad/Desktop/test-websocket-c2-port5000-003.pcap')


import time
import threading
import scapy.all as scapy
from scapy.layers.inet import IP, TCP
from collections import defaultdict

class Sniffer:
    
    def __init__(self, interface="lo", protocol="tcp", port=5000, idle_timeout=30, active_timeout=300):
        self.config = {
            "interface": interface,
            "protocol": protocol,
            "port": port
        }
        self.seen_packets = set()  # This will hold a set of unique packet identifiers

        self.flows = defaultdict(lambda: {
            "src2dst_packets": 0, "src2dst_bytes": 0, "src2dst_max_ps": 0,
            "dst2src_packets": 0, "dst2src_bytes": 0, "dst2src_max_ps": 0
        })  # Store bidirectional flow stats
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.running = False
        self.lock = threading.Lock()
        self.src = 0
        self.dst = 0
        self.packet = 0

    def packet_callback(self, packet):
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return  # Ignore non-TCP packets
        packet_id = (packet[IP].src, packet[IP].dst, packet[TCP].sport, packet[TCP].dport, packet[TCP].seq)
        if packet_id in self.seen_packets:
             print("duplicated packets")
             return
        self.packet += 1
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        packet_size = len(packet)
        with self.lock: # Ensure thread-safe operations
            if (src_ip, src_port, dst_ip, dst_port) in self.flows:
                self.src +=1 
                self.flows[(src_ip, src_port, dst_ip, dst_port)]['src2dst_packets'] += 1
                self.flows[(src_ip, src_port, dst_ip, dst_port)]["src2dst_bytes"] += packet_size
                self.flows[(src_ip, src_port, dst_ip, dst_port)]["src2dst_max_ps"] = max(self.flows[(src_ip, src_port, dst_ip, dst_port)]["src2dst_max_ps"], packet_size)
            elif (dst_ip, dst_port, src_ip, src_port) in self.flows:
                self.dst +=1
                self.flows[(dst_ip, dst_port, src_ip, src_port)]['dst2src_packets'] += 1
                self.flows[(dst_ip, dst_port, src_ip, src_port)]["dst2src_bytes"] += packet_size 
                self.flows[(dst_ip, dst_port, src_ip, src_port)]["dst2src_max_ps"] = max(self.flows[(dst_ip, dst_port, src_ip, src_port)]["dst2src_max_ps"], packet_size)
            else:
                self.flows[(src_ip, src_port, dst_ip, dst_port)] = {'src2dst_packets': 1,"src2dst_bytes":packet_size,"src2dst_max_ps":packet_size,'dst2src_packets': 0,"dst2src_bytes":0,"dst2src_max_ps":0}  

    def run_sniffing(self):
        filter_rule = f"tcp and host 127.0.0.1 and port {self.config['port']}"
        print(f"Sniffing started with filter: {filter_rule} on {self.config['interface']}")
        scapy.sniff(iface=self.config["interface"], filter=filter_rule,
                    prn=self.packet_callback, store=False)

    def print_statistics(self):
        while self.running:
            time.sleep(5)  # Print every 5 seconds
            with self.lock:
                print("\n--- Live Traffic Statistics ---")
                for flow_key, stats in self.flows.items():
                    print(f"Flow {flow_key}:")
                    print(f"  ↳ Sent: {stats['src2dst_packets']} packets, {stats['src2dst_bytes']} bytes (Max PS: {stats['src2dst_max_ps']})")
                    print(f"  ↳ Received: {stats['dst2src_packets']} packets, {stats['dst2src_bytes']} bytes (Max PS: {stats['dst2src_max_ps']})")
                    print(f"===============> src :{self.src} dst: {self.dst} total packet {self.packet}")

    def start(self):
        if self.running:
            print("Sniffer is already running.")
            return

        self.running = True
        threading.Thread(target=self.run_sniffing, daemon=True).start()
        threading.Thread(target=self.print_statistics, daemon=True).start()

        print(f"Packet sniffer started on interface '{self.config['interface']}' monitoring {self.config['protocol'].upper()} port {self.config['port']}")

    def stop(self):
        self.running = False
        print("\nSniffer stopped. Final statistics:")
        self.print_statistics()

if __name__ == "__main__":
    sniffer = Sniffer(interface="lo", port=5000)  # Change port as needed
    sniffer.start()

    try:
        while True:
            time.sleep(1)  # Keep running
    except KeyboardInterrupt:
        sniffer.stop()
"""
import time
import threading
import scapy.all as scapy
from scapy.layers.inet import IP, TCP
from collections import defaultdict

class Sniffer:
    
    def __init__(self, interface="lo", protocol="tcp", port=5000, idle_timeout=30, active_timeout=300):
        self.config = {
            "interface": interface,
            "protocol": protocol,
            "port": port
        }
        self.seen_packets = set()  # Store seen packet identifiers

        self.flows = defaultdict(lambda: {
            "src2dst_packets": 0, "src2dst_bytes": 0, "src2dst_max_ps": 0,
            "dst2src_packets": 0, "dst2src_bytes": 0, "dst2src_max_ps": 0
        })  # Store bidirectional flow stats
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.running = False
        self.lock = threading.Lock()
        self.src = 0
        self.dst = 0
        self.packet = 0
        self.ack = 0
    def packet_callback(self, packet):
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return  # Ignore non-TCP packets

        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        # Unique identifier: (src, dst, sport, dport, seq, ack)
        packet_id = (ip_layer.src, ip_layer.dst, tcp_layer.sport, tcp_layer.dport, tcp_layer.seq, tcp_layer.ack)

        with self.lock:  
            if packet_id in self.seen_packets:
                return  # Ignore exact duplicate retransmissions
            self.seen_packets.add(packet_id)

            self.packet += 1
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            packet_size = len(packet)
            if tcp_layer.flags == "A":
                self.ack +=1
            # Count every TCP packet, including pure ACKs
            if (src_ip, src_port, dst_ip, dst_port) in self.flows:
                self.src += 1
                self.flows[(src_ip, src_port, dst_ip, dst_port)]['src2dst_packets'] += 1
                self.flows[(src_ip, src_port, dst_ip, dst_port)]["src2dst_bytes"] += packet_size
                self.flows[(src_ip, src_port, dst_ip, dst_port)]["src2dst_max_ps"] = max(self.flows[(src_ip, src_port, dst_ip, dst_port)]["src2dst_max_ps"], packet_size)
            elif (dst_ip, dst_port, src_ip, src_port) in self.flows:
                self.dst += 1
                self.flows[(dst_ip, dst_port, src_ip, src_port)]['dst2src_packets'] += 1
                self.flows[(dst_ip, dst_port, src_ip, src_port)]["dst2src_bytes"] += packet_size 
                self.flows[(dst_ip, dst_port, src_ip, src_port)]["dst2src_max_ps"] = max(self.flows[(dst_ip, dst_port, src_ip, src_port)]["dst2src_max_ps"], packet_size)
            else:
                self.flows[(src_ip, src_port, dst_ip, dst_port)] = {'src2dst_packets': 1, "src2dst_bytes": packet_size, "src2dst_max_ps": packet_size, 'dst2src_packets': 0, "dst2src_bytes": 0, "dst2src_max_ps": 0}  
    def run_sniffing(self):
        filter_rule = f"tcp and host 127.0.0.1 and port {self.config['port']}"
        print(f"Sniffing started with filter: {filter_rule} on {self.config['interface']}")
        scapy.sniff(iface=self.config["interface"], filter=filter_rule,
                    prn=self.packet_callback, store=False)

    def print_statistics(self):
        while self.running:
            time.sleep(5)  # Print every 5 seconds
            with self.lock:
                print("\n--- Live Traffic Statistics ---")
                for flow_key, stats in self.flows.items():
                    print(f"Flow {flow_key}:")
                    print(f"  ↳ Sent: {stats['src2dst_packets']} packets, {stats['src2dst_bytes']} bytes (Max PS: {stats['src2dst_max_ps']})")
                    print(f"  ↳ Received: {stats['dst2src_packets']} packets, {stats['dst2src_bytes']} bytes (Max PS: {stats['dst2src_max_ps']})")
                    print(f"===============> src :{self.src} dst: {self.dst} total packet {self.packet} ack: {self.ack}")

    def start(self):
        if self.running:
            print("Sniffer is already running.")
            return

        self.running = True
        threading.Thread(target=self.run_sniffing, daemon=True).start()
        threading.Thread(target=self.print_statistics, daemon=True).start()

        print(f"Packet sniffer started on interface '{self.config['interface']}' monitoring {self.config['protocol'].upper()} port {self.config['port']}")

    def stop(self):
        self.running = False
        print("\nSniffer stopped. Final statistics:")
        self.print_statistics()

if __name__ == "__main__":
    sniffer = Sniffer(interface="lo", port=5000)  # Change port as needed
    sniffer.start()

    try:
        while True:
            time.sleep(1)  # Keep running
    except KeyboardInterrupt:
        sniffer.stop()