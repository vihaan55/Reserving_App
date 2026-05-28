import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Insurance Reserving Model")
st.markdown("**Chain Ladder + Bornhuetter-Ferguson | 50/50 Blend**")

# Sidebar
st.sidebar.header("Parameters")
expected_lr = st.sidebar.number_input("Expected Loss Ratio", 0.0, 1.0, 0.65, 0.01)
valuation_lag = st.sidebar.selectbox("Valuation Lag", [12, 24, 36], index=1)

# Data Input
st.header("Data Input")

tab1, tab2 = st.tabs(["📁 Upload File", "✏️ Manual Input"])

# Tab 1: File Upload
with tab1:
    st.markdown("Upload file with columns: Accident_Year, Premium, Lag_0, Lag_3,...")
    
    uploaded = st.file_uploader("Upload File", type=['xlsx', 'xls', 'csv'])
    
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            
            st.subheader("Preview")
            st.dataframe(df.head(), width='stretch')
            
            # Clean column names - replace spaces with underscores
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('\n', '').str.replace('\r', '')
            
            # Standardize AY column
            for c in df.columns:
                if 'accident' in c.lower() or c.lower() in ['ay', 'year']:
                    df = df.rename(columns={c: 'Accident_Year'})
            
            # Find lag columns (more flexible matching)
            lag_cols = []
            for c in df.columns:
                c_lower = c.lower()
                if 'lag' in c_lower:
                    # Extract numbers from column name
                    nums = ''.join(filter(str.isdigit, c))
                    if nums:
                        lag_cols.append((c, int(nums)))
            
            lag_cols.sort(key=lambda x: x[1])
            lag_cols = [c[0] for c in lag_cols]
            
            # Keep columns
            keep = ['Accident_Year']
            if 'Premium' in df.columns:
                keep.append('Premium')
            keep.extend(lag_cols)
            
            st.session_state.triangle = df[keep].copy()
            st.success(f"✓ Loaded {len(df)} accident years!")
            st.write("Columns found:", st.session_state.triangle.columns.tolist())
            
        except Exception as e:
            st.error(f"Error: {e}")

# Tab 2: Manual
with tab2:
    ay_input = st.text_input("Accident Years", "2021, 2022, 2023")
    prem_input = st.text_input("Premium ($)", "1200000, 1500000, 1800000")
    
    try:
        ays = [int(x.strip()) for x in ay_input.split(',')]
        prems = [float(x.strip()) for x in prem_input.split(',')]
    except:
        ays, prems = [2023], [1500000]
    
    lags = [l for l in [0,3,6,12,24,36] if l <= valuation_lag]
    
    if 'triangle' not in st.session_state:
        sample = [
            [2021, 1200000, 50000, 120000, 200000, 300000, 350000],
            [2022, 1500000, 80000, 180000, 280000, 380000, 0],
            [2023, 1800000, 100000, 220000, 0, 0, 0]
        ]
        cols = ['Accident_Year', 'Premium'] + [f'Lag_{l}' for l in lags]
        st.session_state.triangle = pd.DataFrame(sample[:len(ays)], columns=cols)
    
    triangle_df = st.data_editor(
        st.session_state.triangle,
        column_config={
            "Accident_Year": st.column_config.NumberColumn("Year", disabled=True),
            "Premium": st.column_config.NumberColumn("Premium", format="$"),
            **{f"Lag_{l}": st.column_config.NumberColumn(f"Lag{l}", format="$") for l in lags}
        },
        width='stretch', hide_index=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Year"):
            new_ay = max(ays) + 1
            new_row = pd.DataFrame({
                'Accident_Year': [new_ay],
                'Premium': [1500000],
                **{f'Lag_{l}': [0] for l in lags}
            })
            st.session_state.triangle = pd.concat([st.session_state.triangle, new_row], ignore_index=True)
            st.rerun()
    with col2:
        if st.button("Clear"):
            st.session_state.triangle = pd.DataFrame({'Accident_Year': ays, 'Premium': prems, **{f'Lag_{l}': [0] for l in lags}})
            st.rerun()

# Calculations
def calc_cl(triangle):
    # Find lag columns more robustly
    lag_cols = []
    for c in triangle.columns:
        if 'lag' in c.lower():
            try:
                lag_num = int(''.join(filter(str.isdigit, c)))
                lag_cols.append(lag_num)
            except:
                pass
    
    if not lag_cols:
        return {'reserves': [], 'ultimates': [], 'ldfs': {}, 'error': 'No lag columns found'}
    
    lags = sorted(lag_cols)
    ldfs, reserves, ultimates = {}, [], []
    
    for i in range(len(lags)-1):
        vals = []
        for idx in range(len(triangle)):
            c, n = triangle.loc[idx, f'Lag_{lags[i]}'], triangle.loc[idx, f'Lag_{lags[i+1]}']
            if c>0 and n>0: vals.append(n/c)
        if vals: ldfs[f"{lags[i]}-{lags[i+1]}"] = np.mean(vals)
    
    tail = 1.03
    cf = np.prod(list(ldfs.values())) * tail if ldfs else tail
    
    for idx in range(len(triangle)):
        latest = triangle.loc[idx, f'Lag_{max(lags)}']
        ult = latest * cf
        reserves.append(ult - latest)
        ultimates.append(ult)
    
    return {'reserves': reserves, 'ultimates': ultimates, 'ldfs': ldfs}

def calc_bf(triangle, elr):
    pct = {12: 0.75, 24: 0.92, 36: 0.97}.get(valuation_lag, 0.98)
    reserves, ultimates = [], []
    for idx in range(len(triangle)):
        prem = triangle.loc[idx, 'Premium'] if 'Premium' in triangle.columns else 0
        ult = prem * elr
        reserves.append(ult * (1 - pct))
        ultimates.append(ult)
    return {'reserves': reserves, 'ultimates': ultimates}

# Run
if st.button("🚀 RUN RESERVING MODEL", type="primary", width='stretch'):
    # Check if triangle has data
    if triangle_df.empty:
        st.error("No data! Please upload a file or enter data manually.")
    else:
        cl = calc_cl(triangle_df)
        bf = calc_bf(triangle_df, expected_lr)
        
        # Check for errors
        if 'error' in cl:
            st.error(cl['error'])
            st.write("Available columns:", triangle_df.columns.tolist())
        else:
            blend_r = [(cl['reserves'][i] + bf['reserves'][i])/2 for i in range(len(triangle_df))]
            blend_u = [(cl['ultimates'][i] + bf['ultimates'][i])/2 for i in range(len(triangle_df))]
            
            results = pd.DataFrame({
                'Accident_Year': triangle_df['Accident_Year'],
                'Chain_Ladder': [f"${r:,.0f}" for r in cl['reserves']],
                'BF': [f"${r:,.0f}" for r in bf['reserves']],
                '50_50': [f"${r:,.0f}" for r in blend_r],
                'Ultimate': [f"${u:,.0f}" for u in blend_u]
            })
            
            st.subheader("Results by Accident Year")
            st.dataframe(results, width='stretch')
            
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CL Total", f"${sum(cl['reserves']):,.0f}")
            c2.metric("BF Total", f"${sum(bf['reserves']):,.0f}")
            c3.metric("50/50 Total", f"${sum(blend_r):,.0f}")
            c4.metric("Ultimate", f"${sum(blend_u):,.0f}")
            
            if cl['ldfs']:
                st.subheader("Development Factors")
                ldf_df = pd.DataFrame(list(cl['ldfs'].items()), columns=['Period', 'Factor'])
                st.dataframe(ldf_df, width='stretch')
            
            st.download_button("Download CSV", results.to_csv(index=False), "reserves.csv")