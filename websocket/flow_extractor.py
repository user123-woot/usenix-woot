from nfstream import NFStreamer
import pandas as pd
pcap_addr = "/home/mehrdad/Desktop/test-websocket-c2-port5000-003.pcap"
global ds
my_streamer = NFStreamer(source=pcap_addr,
                        # Disable L7 dissection for readability purpose.
                        n_dissections=0,
                        #idle_timeout= 600,
                        #active_timeout= 1800, 
                        accounting_mode= 3, 
                        statistical_analysis=True)
ds = my_streamer.to_pandas(columns_to_anonymize=[])
col = ["src2dst_packets","src2dst_bytes", "src2dst_max_ps", 
         "dst2src_packets", "dst2src_bytes","dst2src_max_ps"]
for item in col:
    print( f"Column {item}:{ds[item]}")
"""
if pd.DataFrame(ds).shape[0] > 0:
    print("CSV File Shape: {}".format(pd.DataFrame(ds).shape))

    ds.to_csv("websocket/pcap2csv.csv",index=False)
else:
    print (f"This DS is empty, check the pcap file\n{pcap_addr}")
"""