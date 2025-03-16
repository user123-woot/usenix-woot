This only part of the dataset due to lack of space. We intended to uploade the original PCAP files where researcher can benefit.
We will uploade them in a proper repository after acceptance.

PCAP Info ===> folder "dataset_in_PCAP"

This folder contains PCAP files for three C2 frameworks: DeepRed, Caldera, and Mythic. These files capture C2 agents connecting to their respective C2 servers while also engaging in benign activities such as Skype chat, web browsing, video streaming, and DNS queries. As a result, the traffic is a mix of both benign and malicious activity.

To distinguish malicious traffic within each PCAP file, you can use the following port numbers to filter C2-related connections:

Caldera C2 Server: 10.11.54.37:8888
Mythic C2 Server: 10.11.54.67:80
DeepRed C2 Server: 10.11.54.137:5000
