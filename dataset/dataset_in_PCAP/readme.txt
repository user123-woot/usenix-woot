This folder contains sample PCAP files for three C2 frameworks: DeepRed, Caldera, and Mythic. These files capture C2 agents connecting to their respective C2 servers. In addition, there is a sample benign traffic including legitimate activities such as Skype chat, web browsing, video streaming, and DNS queries.

To distinguish malicious traffic within each PCAP file, you can use the following port numbers to filter C2-related connections:

Caldera C2 Server: 10.11.54.37:8888
Mythic C2 Server: 10.11.54.67:80
DeepRed C2 Server: 10.11.54.137:5000
