# -----------------------------------------------------------------------------
# REPLACE THESE FUNCTIONS IN YOUR MAIN STREAMLIT FILE
# -----------------------------------------------------------------------------

# 1. Replace your classify_concentration function with this:
def classify_concentration(row: Dict[str, float], window: str) -> Tuple[int, str]:
    """Wrapper to maintain compatibility with existing code."""
    classifier = EnhancedAirQualityClassifier()
    result = classifier.classify(row, window)
    return (result['cluster_id'], result['cluster_name'])

# 2. Replace your simulate_scenario function with this:
def simulate_scenario(window: str, input_values, features) -> dict:
    """Enhanced simulation using fuzzy classification."""
    from fuzzy_air_classifier import simulate_scenario_enhanced
    return simulate_scenario_enhanced(window, input_values, features)

# 3. Replace your render_temporal_analysis function with this enhanced version:
def render_temporal_analysis_enhanced(df: pd.DataFrame, windows: List[str]) -> None:
    st.title("Análisis temporal: Clasificación híbrida (exacta + fuzzy)")
    st.markdown("Sistema mejorado con lógica difusa para mayor cobertura de datos.")

    selected_window = st.selectbox("Selecciona ventana horaria:", windows, index=0)
    wdf = df[df['time_window'] == selected_window].copy()

    if wdf.empty:
        st.warning("No hay datos para esta ventana.")
        return

    # Initialize enhanced classifier
    classifier = EnhancedAirQualityClassifier()
    
    # Classify all rows and collect results
    classification_results = []
    by_class = {}
    
    for _, row in wdf.iterrows():
        values = row.to_dict()
        result = classifier.classify(values, selected_window)
        
        classification_results.append({
            'cluster_id': result['cluster_id'],
            'cluster_name': result['cluster_name'],
            'match_type': result['match_type'],
            'membership_score': result['membership_score'],
            'row_data': row
        })
        
        cluster_id = result['cluster_id']
        if cluster_id not in by_class:
            by_class[cluster_id] = {
                'name': result['cluster_name'],
                'exact_rows': [],
                'fuzzy_rows': [],
                'exact_count': 0,
                'fuzzy_count': 0
            }
        
        if result['match_type'] == 'exact':
            by_class[cluster_id]['exact_rows'].append(row)
            by_class[cluster_id]['exact_count'] += 1
        elif result['match_type'] == 'fuzzy':
            by_class[cluster_id]['fuzzy_rows'].append(row)
            by_class[cluster_id]['fuzzy_count'] += 1

    # Calculate coverage statistics
    total_rows = len(classification_results)
    exact_count = sum(1 for r in classification_results if r['match_type'] == 'exact')
    fuzzy_count = sum(1 for r in classification_results if r['match_type'] == 'fuzzy')
    unclassified = sum(1 for r in classification_results if r['match_type'] == 'none')
    
    coverage_rate = ((exact_count + fuzzy_count) / total_rows * 100) if total_rows > 0 else 0

    # Display coverage metrics
    st.subheader("📊 Cobertura del sistema de clasificación")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Cobertura total", f"{coverage_rate:.1f}%", 
                "Significativo" if coverage_rate > 70 else "Mejorar")
    col2.metric("Coincidencias exactas", f"{exact_count} ({exact_count/total_rows*100:.1f}%)")
    col3.metric("Coincidencias fuzzy", f"{fuzzy_count} ({fuzzy_count/total_rows*100:.1f}%)")
    col4.metric("Sin clasificar", f"{unclassified} ({unclassified/total_rows*100:.1f}%)")

    if coverage_rate < 50:
        st.warning("⚠️ Cobertura baja. Considera ajustar tolerancias o agregar nuevos clústers.")
    elif coverage_rate > 80:
        st.success("✅ Excelente cobertura del sistema de clasificación.")

    # Distribution visualization
    if by_class:
        # Prepare data for visualization
        viz_data = []
        for cluster_id, info in by_class.items():
            if cluster_id == -1:
                continue  # Skip unclassified for main chart
            viz_data.append({
                'cluster_name': info['name'][:30] + "..." if len(info['name']) > 30 else info['name'],
                'exact_count': info['exact_count'],
                'fuzzy_count': info['fuzzy_count'],
                'total_count': info['exact_count'] + info['fuzzy_count'],
                'cluster_id': cluster_id
            })
        
        if viz_data:
            viz_df = pd.DataFrame(viz_data)
            
            # Stacked bar chart showing exact vs fuzzy matches
            fig_distribution = go.Figure()
            fig_distribution.add_trace(go.Bar(
                name='Exactas',
                x=viz_df['cluster_name'],
                y=viz_df['exact_count'],
                marker_color='darkgreen',
                hovertemplate='<b>%{x}</b><br>Exactas: %{y}<extra></extra>'
            ))
            fig_distribution.add_trace(go.Bar(
                name='Fuzzy',
                x=viz_df['cluster_name'],
                y=viz_df['fuzzy_count'],
                marker_color='lightgreen',
                hovertemplate='<b>%{x}</b><br>Fuzzy: %{y}<extra></extra>'
            ))
            
            fig_distribution.update_layout(
                title="Distribución de clasificaciones por tipo",
                xaxis_title="Categoría",
                yaxis_title="Número de observaciones",
                barmode='stack',
                xaxis_tickangle=-15,
                height=400
            )
            
            st.plotly_chart(fig_distribution, use_container_width=True)

        # Detailed analysis by cluster
        st.subheader("🔍 Análisis detallado por clúster")
        
        for cluster_id in sorted([cid for cid in by_class.keys() if cid != -1]):
            info = by_class[cluster_id]
            total_cluster = info['exact_count'] + info['fuzzy_count']
            
            if total_cluster == 0:
                continue
                
            with st.expander(f"Clúster {cluster_id}: {info['name']} ({total_cluster} observaciones)"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Composición:**")
                    st.write(f"- Exactas: {info['exact_count']}")
                    st.write(f"- Fuzzy: {info['fuzzy_count']}")
                    exact_pct = (info['exact_count'] / total_cluster * 100) if total_cluster > 0 else 0
                    st.write(f"- % Exactas: {exact_pct:.1f}%")
                
                with col2:
                    # Calculate median pollutant levels for this cluster
                    all_rows = info['exact_rows'] + info['fuzzy_rows']
                    if all_rows:
                        pollutants = [c for c in ['CO','NO','NO2','NOX','O3','PM10','PM2.5','SO2'] 
                                    if c in all_rows[0].index]
                        
                        medians = {}
                        for pol in pollutants:
                            values = [row[pol] for row in all_rows if pd.notna(row[pol])]
                            if values:
                                medians[pol] = np.median(values)
                        
                        if medians:
                            st.write("**Niveles típicos (medianas):**")
                            med_text = ", ".join([f"{pol}: {val:.2f}" for pol, val in medians.items()])
                            st.write(med_text)

    # Show unclassified data summary
    if -1 in by_class and by_class[-1]['fuzzy_count'] > 0:
        st.subheader("❓ Datos sin clasificar")
        unclass_count = by_class[-1]['fuzzy_count']
        st.write(f"**{unclass_count} observaciones** no pudieron clasificarse (score < {classifier.min_membership}).")
        
        if st.checkbox("Ver estadísticas de datos sin clasificar"):
            unclass_rows = by_class[-1]['fuzzy_rows']
            if unclass_rows:
                pollutants = [c for c in ['CO','NO','NO2','NOX','O3','PM10','PM2.5','SO2'] 
                            if c in unclass_rows[0].index]
                
                unclass_stats = {}
                for pol in pollutants:
                    values = [row[pol] for row in unclass_rows if pd.notna(row[pol])]
                    if values:
                        unclass_stats[pol] = {
                            'median': np.median(values),
                            'q25': np.percentile(values, 25),
                            'q75': np.percentile(values, 75)
                        }
                
                if unclass_stats:
                    stats_df = pd.DataFrame(unclass_stats).T
                    st.dataframe(stats_df)

# 4. Enhanced simulator function to replace render_simulator:
def render_simulator_enhanced() -> None:
    """Enhanced simulator with fuzzy logic display."""
    st.header("Simulador híbrido: Exacto + Fuzzy")
    st.markdown("Sistema mejorado que muestra coincidencias exactas y aproximadas con scores de pertenencia.")

    # Configuration
    windows = ['morning_peak', 'midday', 'evening_peak', 'night']
    features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
    ranges = {
        'CO': (0.0, 2.0, 0.01), 'NO': (0.0, 0.1, 0.001), 'NO2': (0.0, 0.1, 0.001),
        'NOX': (0.0, 0.2, 0.001), 'O3': (0.0, 0.1, 0.001), 'PM10': (0.0, 150.0, 1.0),
        'PM2.5': (0.0, 75.0, 0.5), 'SO2': (0.0, 0.02, 0.0001)
    }

    # Initialize defaults
    if 'sim_defaults_enhanced' not in st.session_state:
        live = fetch_live_data() or {
            'CO': 0.6, 'NO': 0.03, 'NO2': 0.04, 'NOX': 0.07,
            'O3': 0.05, 'PM10': 45.0, 'PM2.5': 20.0, 'SO2': 0.005
        }
        st.session_state.sim_defaults_enhanced = {k: float(v) for k, v in live.items()}

    selected_window = st.selectbox("Selecciona ventana para simulación:", windows)

    # Initialize values for this window
    if 'sim_values_enhanced' not in st.session_state:
        st.session_state.sim_values_enhanced = {}
    if selected_window not in st.session_state.sim_values_enhanced:
        st.session_state.sim_values_enhanced[selected_window] = dict(st.session_state.sim_defaults_enhanced)

    # Handle pending updates (for buttons)
    pending = st.session_state.pop("sim_pending_enhanced", None)
    if pending and pending.get("window") == selected_window:
        for f, v in pending["values"].items():
            st.session_state.sim_values_enhanced[selected_window][f] = float(v)
            st.session_state[f"slider_enh_{selected_window}_{f}"] = float(v)

    # Create sliders
    st.subheader("Configura niveles de contaminantes")
    col1, col2 = st.columns(2)
    
    for i, f in enumerate(features):
        mn, mx, step = ranges[f]
        current = st.session_state.sim_values_enhanced[selected_window][f]
        key = f"slider_enh_{selected_window}_{f}"
        
        with (col1 if i < 4 else col2):
            if key in st.session_state:
                new_val = st.slider(f, min_value=float(mn), max_value=float(mx), 
                                  step=float(step), key=key)
            else:
                new_val = st.slider(f, min_value=float(mn), max_value=float(mx), 
                                  step=float(step), value=float(current), key=key)
        
        st.session_state.sim_values_enhanced[selected_window][f] = float(new_val)

    # Buttons
    b1, b2, b3 = st.columns([1, 1, 1])
    simulate = b1.button("🎯 Clasificar")
    random_btn = b2.button("🎲 Aleatorio") 
    reset = b3.button("↩ Restablecer")

    # Button actions
    def _rand_on_step(mn, mx, step):
        if step >= 1:
            return float(np.random.randint(int(mn), int(mx) + 1))
        n_steps = int(round((mx - mn) / step))
        return float(mn + np.random.randint(0, n_steps + 1) * step)

    if random_btn:
        new_vals = {f: _rand_on_step(*ranges[f]) for f in features}
        st.session_state["sim_pending_enhanced"] = {"window": selected_window, "values": new_vals}
        st.rerun()

    if reset:
        base_vals = {f: float(st.session_state.sim_defaults_enhanced[f]) for f in features}
        st.session_state["sim_pending_enhanced"] = {"window": selected_window, "values": base_vals}
        st.rerun()

    # Simulation
    if simulate:
        vals = {f: st.session_state.sim_values_enhanced[selected_window][f] for f in features}
        
        # Use enhanced classification
        from fuzzy_air_classifier import EnhancedAirQualityClassifier, calculate_ratios
        classifier = EnhancedAirQualityClassifier()
        result = classifier.classify(vals, selected_window)
        
        # Display results
        st.success("🎯 Clasificación completada!")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clúster", f"{result['cluster_id']}" if result['cluster_id'] != -1 else "Sin clasificar")
        col2.metric("Tipo de match", result['match_type'].title())
        col3.metric("Score de pertenencia", f"{result['membership_score']:.3f}")
        col4.metric("Ventana", selected_window.replace('_', ' '))

        # Match type indicator
        if result['match_type'] == 'exact':
            st.success("✅ **Coincidencia exacta** - Los valores caen perfectamente dentro de los rangos establecidos.")
        elif result['match_type'] == 'fuzzy':
            st.info(f"🎯 **Coincidencia fuzzy** - Los valores están dentro de las tolerancias (score: {result['membership_score']:.3f}).")
        else:
            st.warning(f"❓ **Sin clasificación** - Los valores no se ajustan a ningún patrón conocido (mejor score: {result['membership_score']:.3f}).")

        # Show calculated ratios
        ratios = result['ratios_used']
        if ratios:
            st.subheader("📊 Indicadores auxiliares calculados")
            ratio_cols = st.columns(len(ratios))
            
            ratio_descriptions = {
                'traffic_ratio': ('Ratio de tráfico', 'NO/(NO+NO2)', 'Emisiones frescas vs envejecidas'),
                'photochem_ratio': ('Ratio fotoquímico', 'O3/(O3+NO2)', 'Formación de ozono'),
                'coarse_pm_ratio': ('Ratio PM grueso', 'PM10/PM2.5', 'Polvo vs partículas finas'),
                'fine_pm_ratio': ('Ratio PM', 'PM10/PM2.5', 'Composición de partículas')
            }
            
            for i, (ratio_name, ratio_value) in enumerate(ratios.items()):
                if np.isfinite(ratio_value) and i < len(ratio_cols):
                    desc = ratio_descriptions.get(ratio_name, (ratio_name, '', ''))
                    ratio_cols[i].metric(
                        desc[0], 
                        f"{ratio_value:.3f}",
                        desc[1]
                    )

        # All cluster scores
        st.subheader("🎯 Scores de todos los clústers")
        scores_data = []
        for cluster_id, score_info in result['all_scores'].items():
            scores_data.append({
                'Clúster': cluster_id,
                'Nombre': score_info['name'][:40] + "..." if len(score_info['name']) > 40 else score_info['name'],
                'Score Fuzzy': f"{score_info['score']:.3f}",
                'Match Exacto': '✅' if score_info['exact'] else '❌'
            })
        
        scores_df = pd.DataFrame(scores_data).sort_values('Score Fuzzy', ascending=False)
        st.dataframe(scores_df, use_container_width=True)

        # Enhanced interpretation
        from fuzzy_air_classifier import generate_interpretation_enhanced
        interpretation = generate_interpretation_enhanced(selected_window, vals, result['cluster_name'], result)
        st.subheader("💡 Interpretación detallada")
        st.markdown(interpretation)

        # Input values visualization
        st.subheader("📈 Tu perfil de entrada")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Valores ingresados',
            x=features,
            y=[vals[f] for f in features],
            marker_color='steelblue',
            hovertemplate='<b>%{x}</b><br>Valor: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Perfil de contaminantes ingresado",
            xaxis_title="Contaminantes",
            yaxis_title="Concentración",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# 5. Update the main function to use enhanced versions:
def main_enhanced():
    """Enhanced main function using fuzzy classification."""
    # ... (keep existing setup code) ...
    
    # In the router section, replace:
    if section == "Análisis Temporal":
        render_temporal_analysis_enhanced(fdf, windows)  # Use enhanced version
    elif section == "Simulador":
        render_simulator_enhanced()  # Use enhanced version

# 6. Import the fuzzy classifier at the top of your file:
# Add this import at the top of your main Streamlit file:
# from fuzzy_air_classifier import (
#     EnhancedAirQualityClassifier, 
#     classify_concentration_enhanced, 
#     simulate_scenario_enhanced,
#     calculate_ratios,
#     analyze_classification_coverage
# )