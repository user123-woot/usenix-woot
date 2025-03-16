This folder contains traffic flows from DeppRed C2, where various C2 agents are configured to execute random remote commands. 
These commands include those commonly used by APT groups for network discovery, account discovery, file and directory discovery, system information discovery, and system service discovery, as defined by the MITRE ATT&CK framework. 
Additionally, the agents exfiltrate random files, including JPEGs, PDFs, MP4s, and DOCX documents, from the victim.

The CSV files represent traffic flows extracted from the original PCAP files using NFStream, as discussed in the paper. All CSV files are labeled, though they are primarily used for generating malicious traffic, which will be utilized for GAN training and the development of various ML-based NIDS.
