# DeepRed C2 Framework
This repository contains the files used to develop the C2 framework prposed for the USENIX WOOT 2025. To run the C2 framework, navigate to the "websock_c2" directory, start the c2-sniffer to launch the server, and then connect the bot to it.
It is important to ensure all required dependencies are installed beforehand, and to configure the IP addresses for both the C2 server and the bot.

### DeepRed C2 Demo
The GIF below demonstrates the DeepRed C2 framework in action. On the left side, the C2 server code is running on a VM within a controlled environment. On the right side, another VM hosts the C2 agent code (bot.py), which initiates a WebSocket connection to the C2 server. The agent then autonomously executes random remote commands and performs data exfiltration. Throughout the process, all network traffic is captured using tcpdump.

![2025-03-16 19-48-32](https://github.com/user-attachments/assets/448b5ca4-2849-40af-b2da-41f780ce298a)


# Sample Proposed Dataset
In the "dataset" folder, we have included a subset of the dataset in CSV format, along with sample benign and C2 raw traffic in PCAP files. The full dataset is too large to be included here. Once the paper is accepted, the original PCAP files will be uploaded to the appropriate repository.
Malicious IP:PORT
- Caldera C2 Server: 10.11.54.37:8888
- Mythic C2 Server: 10.11.54.67:80
- DeepRed C2 Server: 10.11.54.137:5000

### Dataset folder structure  
```
/dataset/dataset_in_PCAP/
   |_____caldera-C2-over-http-sample-packets.pcap
   |_____mythic-Athena-agent-for-C2-over-http-sample-packets.pcap
   |_____sample deepred websocket.pcap
   |_____sample-benign-traffic-browse-skype-etc.pcap`
/dataset/deepred-auto-c2/
   |_____deepred_autoc2_labeled_1-3-2025-001.csv
   |_____deepred_autoc2_labeled_1-3-2025.csv
/dataset/
   |_____deepred_autoc2_labeled_1-3-2025.csv
   |_____nids-generalization_test_being and malicious mixed.zip
```
