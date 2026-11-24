pathology_type, organic_rx, chem_rx = get_kvk_advisory(raw_label)
            st.markdown(f"#### Recommended Field Care ({pathology_type})")
            c_org, c_chem = st.columns(2)
            with c_org:
                st.markdown(f"""
                <div class="remedy-organic">
                    <div class="remedy-head" style="color: #1E4620;">Option 1: जैविक उपाय (Organic / Bio-Control)</div>
                    <p class="remedy-text">{organic_rx}</p>
                </div>
                """, unsafe_allow_html=True)
            with c_chem:
                st.markdown(f"""
                <div class="remedy-chemical">
                    <div class="remedy-head" style="color: #9A3412;">Option 2: रासायनिक उपचार (KVK Spray Dosage)</div>
                    <p class="remedy-text">{chem_rx}</p>
                </div>
                """, unsafe_allow_html=True)
