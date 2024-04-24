import plotly.graph_objects as go
import pandas as pd

netflow_data  = pd.read_csv("random_nf.csv")
req_cols = ["SrcIPaddress", "DstIPaddress", "SrcP", "DstP", "P", "Pkts", "Octets"]

netflow_ip_data = netflow_data[req_cols].groupby(["SrcIPaddress", "DstIPaddress","SrcP","DstP"]).sum().reset_index()
netflow_ip_data = netflow_ip_data.sample(frac=1, random_state=0)
# netflow_ip_data = netflow_ip_data.nlargest(100, columns=["dpkts"])

hosts = list(set(netflow_ip_data["SrcIPaddress"].to_list() + netflow_ip_data["DstIPaddress"].to_list()+netflow_ip_data["SrcP"].to_list()+netflow_ip_data["DstP"].to_list()))
host_idx_dict = {host: idx for idx,host in enumerate(hosts)}

num_hosts = 30
sources = [host_idx_dict[host] for host in netflow_ip_data["SrcIPaddress"].to_list()[0:num_hosts]]
targets = [host_idx_dict[host] for host in netflow_ip_data["DstIPaddress"].to_list()[0:num_hosts]]
pkt_values =  netflow_ip_data["Pkts"].to_list()[0:num_hosts]
bytes_values =  netflow_ip_data["Octets"].to_list()[0:num_hosts]

.

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = hosts,
      color = "blue"
    ),
    link = dict(
      source = sources,
      target = targets,
      value = pkt_values,
      hovercolor="lightgreen"
  ))])

fig.update_layout(title_text="Sankey Diagram: Number of packets transferred between hosts", font_size=10)
fig.show()


fig1 = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = hosts,
      color = "blue"
    ),
    link = dict(
      source = sources,
      target = targets,
      value = bytes_values,
      hovercolor="lightgreen"
  ))])

fig1.update_layout(title_text="Sankey Diagram: Number of bytes transferred between hosts", font_size=10)
fig1.show()
