import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import community.community_louvain as community_louvain
import pandas as pd

def setup_directories(folders=["img"]):
    """Vytvorí potrebné priečinky, ak neexistujú."""
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def load_data(filepath):
    """Načíta graf z edgelist súboru."""
    print(f"Načítavam dáta z: {filepath}")
    G = nx.read_edgelist(filepath, nodetype=int)
    print(f"-> Uzlov: {G.number_of_nodes()}")
    print(f"-> Hrán: {G.number_of_edges()}")
    return G

def communities_to_partition(communities):
    """
    Pomocná funkcia: Prevedie zoznam množín (výstup LPA/Greedy) 
    na slovník {uzol: id_komunity} (formát pre Louvain/Modularitu).
    """
    partition = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            partition[node] = idx
    return partition

def calculate_modularity(G, partition):
    """Vypočíta modularitu pre dané rozdelenie."""
    return community_louvain.modularity(partition, G)

def draw_graph_raw(G, pos, filename="img/data.png"):
    """Vykreslí a uloží surové dáta bez komunít."""
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='black')
    nx.draw_networkx_edges(G, pos, alpha=0.05)
    plt.title("Základná štruktúra siete", fontsize=15)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def draw_communities(G, partition, pos, title, filename):
    """
    Vykreslí graf ofarbený podľa komunít.
    Opravené 'get_cmap' warningy.
    """
    plt.figure(figsize=(12, 12))
    
    # Získame unikátne ID komunít
    unique_comms = list(set(partition.values()))
    num_comms = len(unique_comms)
    
    # Použitie colormap bez DeprecationWarning
    cmap = plt.colormaps.get_cmap('tab20') 
    
    # Vykreslenie hrán (v pozadí)
    nx.draw_networkx_edges(G, pos, alpha=0.03, edge_color='gray')

    # Vykreslenie uzlov po komunitách
    for i, comm_id in enumerate(unique_comms):
        nodes = [n for n in G.nodes() if partition[n] == comm_id]
        
        # Farba - cyklíme, ak je komunít viac ako farieb v palete
        color = cmap(i % 20)
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_size=20,
            node_color=[color]
        )

    plt.title(f"{title}\n(Počet komunít: {num_comms})", fontsize=15)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_results_table(results_list, filename="img/vyhodnotenie.png"):
    """
    Vytvorí a uloží PNG tabuľku s výsledkami.
    results_list: zoznam slovníkov [{'Model': '...', 'Modularity': ...}, ...]
    """
    df = pd.DataFrame(results_list)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title("Porovnanie výsledkov", fontsize=14, y=1.1)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return df