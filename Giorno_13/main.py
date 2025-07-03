"""
NER Graph Extractor - App Streamlit Standalone
VERSIONE CORRETTA con fix Plotly e grafi interattivi
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime
import json

from config import APP_CONFIG, ENTITY_COLORS, LIMITS
from ner_extractor import SimpleNERExtractor, entities_to_dataframe
from graph_builder import SimpleGraphBuilder, build_knowledge_graph

def setup_page():
    """Configura la pagina Streamlit"""
    st.set_page_config(**APP_CONFIG)
    
    st.title("🕸️ NER Graph Extractor")
    st.markdown("**Estrai entità da documenti e visualizza grafi di conoscenza**")
    st.markdown("---")

def sidebar_info():
    """Mostra informazioni nella sidebar"""
    with st.sidebar:
        st.header("ℹ️ Come Funziona")
        
        st.markdown("""
        **Passaggi:**
        1. 📤 Carica documenti .txt
        2. 🔍 Estrai entità automaticamente
        3. 🕸️ Visualizza grafo di relazioni
        4. 💾 Esporta risultati
        """)
        
        st.markdown("---")
        
        st.header("🎯 Tipi Entità")
        st.markdown("""
        - 👤 **PERSON**: Nomi di persone
        - 🏢 **ORGANIZATION**: Aziende, enti
        - 📍 **LOCATION**: Luoghi, indirizzi
        - 📅 **DATE**: Date e scadenze
        - 💰 **MONEY**: Importi e valute
        - 📧 **EMAIL**: Indirizzi email
        """)
        
        st.markdown("---")
        
        if st.button("🔄 Reset Tutto"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def upload_section():
    """Sezione upload documenti"""
    st.header("📤 Carica Documenti")
    
    uploaded_files = st.file_uploader(
        "Seleziona file di testo (.txt)",
        type=['txt'],
        accept_multiple_files=True,
        help=f"Massimo {LIMITS['max_files']} file, {LIMITS['max_file_size_mb']}MB ciascuno"
    )
    
    if uploaded_files:
        # Valida file
        valid_files = []
        for file in uploaded_files:
            if file.size > LIMITS['max_file_size_mb'] * 1024 * 1024:
                st.error(f"❌ File '{file.name}' troppo grande (max {LIMITS['max_file_size_mb']}MB)")
                continue
            valid_files.append(file)
        
        if len(valid_files) > LIMITS['max_files']:
            st.warning(f"⚠️ Troppi file! Usando i primi {LIMITS['max_files']}")
            valid_files = valid_files[:LIMITS['max_files']]
        
        if valid_files:
            # Leggi contenuto file
            documents = {}
            for file in valid_files:
                try:
                    content = file.read().decode('utf-8')
                    documents[file.name] = content
                except Exception as e:
                    st.error(f"❌ Errore lettura '{file.name}': {e}")
            
            # Salva in session state
            st.session_state.documents = documents
            
            # Mostra anteprima
            st.success(f"✅ Caricati {len(documents)} documenti")
            
            with st.expander("👀 Anteprima Documenti"):
                for filename, content in documents.items():
                    st.subheader(f"📄 {filename}")
                    preview = content[:300] + "..." if len(content) > 300 else content
                    st.text_area(
                        f"Contenuto {filename}:",
                        value=preview,
                        height=150,
                        disabled=True,
                        key=f"preview_{filename}"
                    )
    
    # Mostra documenti caricati
    if 'documents' in st.session_state:
        st.info(f"📊 Documenti pronti: {len(st.session_state.documents)}")
        return True
    
    return False

def extraction_section():
    """Sezione estrazione entità"""
    st.header("🔍 Estrazione Entità NER")
    
    if 'documents' not in st.session_state:
        st.warning("⚠️ Carica prima alcuni documenti")
        return False
    
    # Configurazioni estrazione
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "🎯 Soglia Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Confidence minima per accettare un'entità"
        )
    
    with col2:
        extract_relations = st.checkbox(
            "🔗 Estrai Relazioni",
            value=True,
            help="Cerca relazioni tra entità estratte"
        )
    
    # Bottone estrazione
    if st.button("🚀 Avvia Estrazione", type="primary"):
        
        with st.spinner("🤖 Estraendo entità..."):
            # Inizializza estrattore
            extractor = SimpleNERExtractor()
            
            # Estrai entità
            entities = extractor.extract_from_documents(st.session_state.documents)
            
            # Filtra per confidence
            filtered_entities = [
                e for e in entities 
                if e.confidence >= confidence_threshold
            ]
            
            # Salva risultati
            st.session_state.entities = filtered_entities
            st.session_state.extractor_stats = extractor.get_statistics()
            
            # Costruisci grafo se richiesto
            if extract_relations:
                with st.spinner("🕸️ Costruendo grafo..."):
                    graph_builder = build_knowledge_graph(
                        filtered_entities, 
                        st.session_state.documents
                    )
                    st.session_state.graph_builder = graph_builder
                    st.session_state.graph_stats = graph_builder.get_graph_stats()
        
        st.success("✅ Estrazione completata!")
        st.rerun()
    
    # Mostra risultati se disponibili
    if 'entities' in st.session_state:
        show_extraction_results()
        return True
    
    return False

def show_extraction_results():
    """Mostra risultati estrazione"""
    entities = st.session_state.entities
    stats = st.session_state.extractor_stats
    
    st.subheader("📊 Risultati Estrazione")
    
    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entità Totali", stats.get('total_entities', 0))
    
    with col2:
        st.metric("Tipi Diversi", stats.get('unique_types', 0))
    
    with col3:
        st.metric("Confidence Media", f"{stats.get('avg_confidence', 0):.2f}")
    
    with col4:
        st.metric("Documenti", len(stats.get('by_document', {})))
    
    # Grafici distribuzione
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Grafico tipi entità
        type_counts = stats.get('by_type', {})
        if type_counts:
            fig_pie = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="🏷️ Distribuzione Tipi Entità",
                color_discrete_map=ENTITY_COLORS
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        # Grafico per documento - FIX PLOTLY
        doc_counts = stats.get('by_document', {})
        if doc_counts:
            fig_bar = px.bar(
                x=list(doc_counts.keys()),
                y=list(doc_counts.values()),
                title="📄 Entità per Documento"
            )
            # FIX: Usa update_layout invece di update_xaxis/update_yaxis
            fig_bar.update_layout(
                xaxis_title="Documento",
                yaxis_title="Numero Entità"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Tabella entità
    st.subheader("📋 Entità Estratte")
    
    if entities:
        df = entities_to_dataframe(entities)
        
        # Filtri tabella
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            selected_types = st.multiselect(
                "Filtra per tipo:",
                options=df['Tipo'].unique(),
                default=df['Tipo'].unique(),
                key="entity_type_filter"
            )
        
        with col_filter2:
            min_conf = st.slider(
                "Confidence minima:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="confidence_table_filter"
            )
        
        # Applica filtri
        filtered_df = df[
            (df['Tipo'].isin(selected_types)) & 
            (df['Confidence'] >= min_conf)
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        st.info(f"Visualizzando {len(filtered_df)} entità su {len(df)} totali")

def graph_section():
    """Sezione visualizzazione grafo"""
    st.header("🕸️ Grafo di Conoscenza")
    
    if 'graph_builder' not in st.session_state:
        st.warning("⚠️ Estrai prima le entità con l'opzione 'Estrai Relazioni'")
        return False
    
    graph_builder = st.session_state.graph_builder
    graph_stats = st.session_state.graph_stats
    
    # Statistiche grafo
    st.subheader("📈 Statistiche Grafo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nodi (Entità)", graph_stats.get('nodes', 0))
    
    with col2:
        st.metric("Archi (Relazioni)", graph_stats.get('edges', 0))
    
    with col3:
        st.metric("Densità", f"{graph_stats.get('density', 0):.3f}")
    
    # Distribuzione relazioni
    if 'relation_types' in graph_stats and graph_stats['relation_types']:
        st.subheader("🔗 Tipi di Relazioni")
        
        rel_types = graph_stats['relation_types']
        fig_rel = px.bar(
            x=list(rel_types.keys()),
            y=list(rel_types.values()),
            title="Distribuzione Tipi di Relazioni"
        )
        st.plotly_chart(fig_rel, use_container_width=True)
    
    # NUOVO: Opzioni visualizzazione grafo
    st.subheader("🌐 Opzioni Visualizzazione")
    
    col_viz1, col_viz2, col_viz3 = st.columns(3)
    
    with col_viz1:
        graph_layout = st.selectbox(
            "🎨 Layout Grafo:",
            options=["spring", "circular", "kamada_kawai", "random"],
            index=0,
            help="Algoritmo per posizionare i nodi"
        )
    
    with col_viz2:
        node_size = st.slider(
            "📏 Dimensione Nodi:",
            min_value=8,
            max_value=20,
            value=12,
            help="Dimensione dei nodi nel grafo"
        )
    
    with col_viz3:
        show_labels = st.checkbox(
            "🏷️ Mostra Etichette",
            value=True,
            help="Mostra nomi entità sui nodi"
        )
    
    # Visualizzazione grafo interattiva
    st.subheader("🌐 Visualizzazione Interattiva")
    
    # BOTTONI PER DIVERSI TIPI DI GRAFO
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("📊 Grafo Completo", type="primary"):
            create_graph_visualization(
                graph_builder, 
                layout=graph_layout, 
                node_size=node_size, 
                show_labels=show_labels
            )
    
    with col_btn2:
        if st.button("🎯 Solo Persone & Org"):
            create_filtered_graph(
                graph_builder, 
                filter_types=["PERSON", "ORGANIZATION"],
                layout=graph_layout,
                node_size=node_size,
                show_labels=show_labels
            )
    
    with col_btn3:
        if st.button("💰 Focus Pagamenti"):
            create_filtered_graph(
                graph_builder, 
                filter_types=["PERSON", "ORGANIZATION", "MONEY"],
                layout=graph_layout,
                node_size=node_size,
                show_labels=show_labels
            )
    
    # Tabelle dettagliate
    st.subheader("📋 Dati Dettagliati")
    
    tab_entities, tab_relations = st.tabs(["🏷️ Entità", "🔗 Relazioni"])
    
    with tab_entities:
        entities_df, _ = graph_builder.to_dataframes()
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
        else:
            st.info("Nessuna entità nel grafo")
    
    with tab_relations:
        _, relations_df = graph_builder.to_dataframes()
        if not relations_df.empty:
            st.dataframe(relations_df, use_container_width=True)
        else:
            st.info("Nessuna relazione trovata")
    
    return True

def create_graph_visualization(graph_builder, layout="spring", node_size=12, show_labels=True):
    """Crea visualizzazione interattiva del grafo - VERSIONE CORRETTA"""
    graph = graph_builder.graph
    
    if graph.number_of_nodes() == 0:
        st.warning("Nessun nodo da visualizzare")
        return
    
    # Limita nodi per performance
    if graph.number_of_nodes() > LIMITS['max_nodes_display']:
        st.warning(f"Grafo grande! Visualizzo i {LIMITS['max_nodes_display']} nodi più connessi")
        
        # Seleziona nodi più connessi
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        selected_nodes = [node for node, _ in top_nodes[:LIMITS['max_nodes_display']]]
        graph = graph.subgraph(selected_nodes)
    
    # Layout del grafo - MIGLIORATO
    try:
        if layout == "spring":
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        else:  # random
            pos = nx.random_layout(graph)
    except Exception as e:
        st.warning(f"Errore layout {layout}, uso random")
        pos = nx.random_layout(graph)
    
    # Prepara dati per Plotly
    edge_x, edge_y = [], []
    edge_info = []
    
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Info per hover
        edge_data = edge[2]
        relation_type = edge_data.get('relation_type', 'UNKNOWN')
        edge_info.append(f"{edge[0]} → {edge[1]} ({relation_type})")
    
    # Traccia archi
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Relazioni'
    )
    
    # Traccia nodi
    node_x, node_y, node_text, node_info, node_colors, node_sizes = [], [], [], [], [], []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Informazioni nodo
        attrs = graph.nodes[node]
        entity_type = attrs.get('entity_type', 'OTHER')
        confidence = attrs.get('confidence', 0)
        documents = attrs.get('documents', [])
        connections = graph.degree(node)
        
        # Testo del nodo
        if show_labels:
            display_text = node if len(node) <= 15 else node[:12] + "..."
            node_text.append(display_text)
        else:
            node_text.append("")
        
        # Info hover dettagliate
        node_info.append(
            f"<b>{node}</b><br>"
            f"Tipo: {entity_type}<br>"
            f"Confidence: {confidence:.2f}<br>"
            f"Connessioni: {connections}<br>"
            f"Documenti: {', '.join(documents)}"
        )
        
        # Colore per tipo
        node_colors.append(ENTITY_COLORS.get(entity_type, '#95A5A6'))
        
        # Dimensione proporzionale alle connessioni
        base_size = node_size
        connection_bonus = min(connections * 2, 10)  # Max bonus +10
        node_sizes.append(base_size + connection_bonus)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        hovertext=node_info,
        textposition="middle center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        name='Entità'
    )
    
    # Crea figura - VERSIONE MIGLIORATA
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="🕸️ Grafo di Conoscenza Interattivo",
                x=0.5,
                font_size=16
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="💡 Hover sui nodi per dettagli • Zoom e pan abilitati",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(color="#888", size=12)
                )
            ],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                fixedrange=False  # Abilita zoom
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                fixedrange=False  # Abilita zoom
            ),
            height=600,
            plot_bgcolor='white'
        )
    )
    
    # CONFIGURAZIONE INTERATTIVA
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'knowledge_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'height': 600,
            'width': 800,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Legenda colori MIGLIORATA
    display_color_legend()
    
    # Statistiche del grafo visualizzato
    st.info(f"📊 Visualizzando: {graph.number_of_nodes()} nodi, {graph.number_of_edges()} relazioni")

def create_filtered_graph(graph_builder, filter_types, layout="spring", node_size=12, show_labels=True):
    """Crea grafo filtrato per tipi specifici"""
    graph = graph_builder.graph
    
    # Filtra nodi per tipo
    filtered_nodes = []
    for node, attrs in graph.nodes(data=True):
        entity_type = attrs.get('entity_type', 'OTHER')
        if entity_type in filter_types:
            filtered_nodes.append(node)
    
    if not filtered_nodes:
        st.warning(f"Nessun nodo trovato per i tipi: {', '.join(filter_types)}")
        return
    
    # Crea sottografo
    subgraph = graph.subgraph(filtered_nodes)
    
    # Crea un builder temporaneo per la visualizzazione
    temp_builder = SimpleGraphBuilder()
    temp_builder.graph = subgraph
    
    st.info(f"🎯 Filtro applicato: {', '.join(filter_types)}")
    create_graph_visualization(temp_builder, layout, node_size, show_labels)

def display_color_legend():
    """Mostra legenda colori migliorata"""
    st.subheader("🎨 Legenda Tipi Entità")
    
    # Raggruppa in 2 righe per layout migliore
    entity_types = list(ENTITY_COLORS.items())
    mid_point = len(entity_types) // 2
    
    col1, col2 = st.columns(2)
    
    with col1:
        for entity_type, color in entity_types[:mid_point]:
            st.markdown(
                f'<span style="color: {color}; font-size: 18px;">●</span> '
                f'<b>{entity_type}</b>', 
                unsafe_allow_html=True
            )
    
    with col2:
        for entity_type, color in entity_types[mid_point:]:
            st.markdown(
                f'<span style="color: {color}; font-size: 18px;">●</span> '
                f'<b>{entity_type}</b>', 
                unsafe_allow_html=True
            )

def export_section():
    """Sezione export risultati"""
    st.header("💾 Export Risultati")
    
    if 'entities' not in st.session_state:
        st.warning("⚠️ Nessun risultato da esportare")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Export JSON")
        
        if 'graph_builder' in st.session_state:
            json_data = st.session_state.graph_builder.export_to_json()
            
            st.download_button(
                label="💾 Scarica Grafo JSON",
                data=json_data,
                file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("Estrai relazioni per abilitare export JSON")
    
    with col2:
        st.subheader("📊 Export CSV")
        
        # Export entità
        entities_df = entities_to_dataframe(st.session_state.entities)
        csv_entities = entities_df.to_csv(index=False)
        
        st.download_button(
            label="📋 Scarica Entità CSV",
            data=csv_entities,
            file_name=f"entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Export relazioni se disponibili
        if 'graph_builder' in st.session_state:
            _, relations_df = st.session_state.graph_builder.to_dataframes()
            if not relations_df.empty:
                csv_relations = relations_df.to_csv(index=False)
                
                st.download_button(
                    label="🔗 Scarica Relazioni CSV",
                    data=csv_relations,
                    file_name=f"relations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Funzione principale dell'app"""
    setup_page()
    sidebar_info()
    
    # Workflow principale
    st.markdown("## 🔄 Workflow")
    
    # Step 1: Upload
    with st.container():
        has_documents = upload_section()
    
    st.markdown("---")
    
    # Step 2: Estrazione
    if has_documents:
        with st.container():
            has_entities = extraction_section()
    else:
        st.info("👆 Carica prima alcuni documenti")
        has_entities = False
    
    st.markdown("---")
    
    # Step 3: Grafo
    if has_entities:
        with st.container():
            graph_section()
    else:
        st.info("👆 Estrai prima le entità")
    
    st.markdown("---")
    
    # Step 4: Export
    with st.container():
        export_section()

if __name__ == "__main__":
    main()