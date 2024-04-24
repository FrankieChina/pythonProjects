import plotly.graph_objects as go
import pandas as pd

netflow_data  = pd.read_csv("random_nf.csv")
req_cols = ["SrcIPaddress", "DstIPaddress", "SrcP", "DstP", "P", "Pkts", "Octets"]

df1 = netflow_data[req_cols].groupby(['SrcIPaddress','P']).sum().reset_index()
df1 = df1[['SrcIPaddress', 'P', 'Pkts']]
df1.columns = ['source','target','value']

df2 = netflow_data[req_cols].groupby(['P','DstP']).sum().reset_index()
df2 = df2[['P', 'DstP', 'Pkts']]
df2.columns = ['source','target','value']

df3 = netflow_data[req_cols].groupby(['DstP','DstIPaddress']).sum().reset_index()
df3 = df3[['DstP', 'DstIPaddress', 'Pkts']]
df3.columns = ['source','target','value']

links = pd.concat([df1, df2, df3], axis=0)

unique_source_target = list(pd.unique(links[['source','target']].values.ravel('k')))

mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
#print(mapping_dict)

links['source'] = links['source'].map(mapping_dict)
links['target'] = links['target'].map(mapping_dict)

links_dict = links.to_dict(orient='list')
#print(links_dict)

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = unique_source_target,
      color = "blue"
    ),
    link = dict(
      source = links_dict['source'],
      target = links_dict['target'],
      value = links_dict['value'],
      #hovercolor="lightgreen"
  ))])

fig.update_layout(
    title_text="Sankey Diagram: Number of packets transferred from attacker to destination addresses",
    font_size=10,
    annotations=[
        dict(
            x=0.5,
            y=-0.1,
            xref='paper',
            yref='paper',
            showarrow=False,
            text="Names: Frankie China Quintero, Jaspreet Singh, and Tiffany Kawamura",
            font=dict(
                size=12,
                color='grey'
            )
        )
    ]
)

fig.show()