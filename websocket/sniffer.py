import time
import threading
import scapy.all as scapy
from scapy.layers.inet import IP, TCP

class FlowStats:
    """Represents a bidirectional flow with expiration tracking."""
    def __init__(self, packet):
        self.start_time = time.time()
        self.last_seen = self.start_time
        self.src2dst_packets = 0
        self.dst2src_packets = 0
        self.src2dst_bytes = 0
        self.dst2src_bytes = 0
        self.src2dst_max_ps = 0
        self.dst2src_max_ps = 0
        self.update(packet)

    def update(self, packet):
        """Update flow stats based on a new packet."""
        self.last_seen = time.time()
        packet_size = len(packet)
        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        if tcp_layer.dport == 5000:  # Client request (client → server)
            self.src2dst_packets += 1
            self.src2dst_bytes += packet_size
            self.src2dst_max_ps = max(self.src2dst_max_ps, packet_size)
        elif tcp_layer.sport == 5000:   # Server response (server → client)
            self.dst2src_packets += 1
            self.dst2src_bytes += packet_size
            self.dst2src_max_ps = max(self.dst2src_max_ps, packet_size)

def get_flow_key(src_ip, src_port, dst_ip, dst_port, protocol=6, vlan_id=0, tunnel_id=0):
    """Generate a direction-agnostic flow key."""
    if (src_ip, src_port) < (dst_ip, dst_port):  # Always store flows in sorted order
        return (src_ip, src_port, dst_ip, dst_port, protocol, vlan_id, tunnel_id)
    return (dst_ip, dst_port, src_ip, src_port, protocol, vlan_id, tunnel_id)

class Sniffer:
    """Handles packet tracking with multi-threaded sniffing and real-time stats."""
    def __init__(self, interface="lo", protocol="tcp", port=5000, idle_timeout=30, active_timeout=300):
        self.config = {
            "interface": interface,
            "protocol": protocol,
            "port": port
        }
        self.flows = {}  # Active flow cache
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.running = False
        self.lock = threading.Lock()

    def packet_callback(self, packet):
        """Process incoming packets and update flow statistics."""
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return  # Ignore non-TCP packets

        ip_layer = packet[IP]
        tcp_layer = packet[TCP]
        packet_size = len(packet)

        # Generate a direction-agnostic flow key
        flow_key = get_flow_key(
            ip_layer.src, tcp_layer.sport,
            ip_layer.dst, tcp_layer.dport,
            6,  # TCP protocol
            getattr(packet, "vlan_id", 0),
            getattr(packet, "tunnel_id", 0)
        )

        current_time = time.time()

        with self.lock:  # Ensure thread-safe operations
            # Expire old flows
            expired_flows = [
                key for key, flow in self.flows.items()
                if (current_time - flow.last_seen > self.idle_timeout) or
                (current_time - flow.start_time > self.active_timeout)
            ]
            for key in expired_flows:
                del self.flows[key]

            # Create or update flow
            if flow_key not in self.flows:
                self.flows[flow_key] = FlowStats(packet)

            # Determine direction based on destination port
            if tcp_layer.dport == self.config["port"]:  # Client request (client → server)
                self.flows[flow_key].src2dst_packets += 1
                self.flows[flow_key].src2dst_bytes += packet_size
                self.flows[flow_key].src2dst_max_ps = max(self.flows[flow_key].src2dst_max_ps, packet_size)
            elif tcp_layer.sport == self.config["port"]:  # Server response (server → client)
                self.flows[flow_key].dst2src_packets += 1
                self.flows[flow_key].dst2src_bytes += packet_size
                self.flows[flow_key].dst2src_max_ps = max(self.flows[flow_key].dst2src_max_ps, packet_size)

    def run_sniffing(self):
        """Start sniffing packets in a separate thread."""
        filter_rule = f"tcp and host 127.0.0.1 and port {self.config['port']}"
        print(f"Sniffing started with filter: {filter_rule} on {self.config['interface']}")
        scapy.sniff(iface=self.config["interface"], filter=filter_rule,
                    prn=self.packet_callback, store=False)

    def print_statistics(self):
        """Continuously print traffic statistics while running."""
        while self.running:
            time.sleep(5)  # Print every 5 seconds
            with self.lock:
                print("\n--- Live Traffic Statistics ---")
                for flow_key, stats in self.flows.items():
                    print(f"Flow {flow_key}:")
                    print(f"  ↳ Sent: {stats.src2dst_packets} packets, {stats.src2dst_bytes} bytes (Max PS: {stats.src2dst_max_ps})")
                    print(f"  ↳ Received: {stats.dst2src_packets} packets, {stats.dst2src_bytes} bytes (Max PS: {stats.dst2src_max_ps})")

    def start(self):
        """Start packet sniffing in a separate thread."""
        if self.running:
            print("Sniffer is already running.")
            return

        self.running = True
        threading.Thread(target=self.run_sniffing, daemon=True).start()
        threading.Thread(target=self.print_statistics, daemon=True).start()

        print(f"Packet sniffer started on interface '{self.config['interface']}' monitoring {self.config['protocol'].upper()} port {self.config['port']}")

    def stop(self):
        """Stop sniffing."""
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