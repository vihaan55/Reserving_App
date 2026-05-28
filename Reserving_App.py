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
    st.markdown("Upload: Accident_Year, Development_Lag, Amount")
    
    uploaded = st.file_uploader("Upload File", type=['xlsx', 'xls', 'csv'])
    
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            
            st.subheader("Raw Data (First 10 rows)")
            st.dataframe(df.head(10), width='stretch')
            
            # Find columns
            ay_col, lag_col, amount_col = None, None, None
            
            for c in df.columns:
                c_lower = c.lower()
                if 'accident' in c_lower or c_lower in ['year', 'ay']:
                    ay_col = c
                elif 'development' in c_lower or 'lag' in c_lower:
                    lag_col = c
                elif 'amount' in c_lower or 'claim' in c_lower:
                    amount_col = c
            
            if ay_col and lag_col and amount_col:
                # Aggregate amounts by AY and Lag
                grouped = df.groupby([ay_col, lag_col])[amount_col].sum().reset_index()
                
                # Pivot to create triangle
                triangle = grouped.pivot(index=ay_col, columns=lag_col, values=amount_col).fillna(0)
                triangle = triangle.reset_index()
                triangle = triangle.rename(columns={ay_col: 'Accident_Year'})
                
                # Add all lag columns
                all_lags = [0, 3, 6, 12, 24, 36, 48]
                for lag in all_lags:
                    if lag not in triangle.columns:
                        triangle[lag] = 0
                
                # Select needed columns
                premium_cols = ['Accident_Year']
                premium_cols += [l for l in all_lags if l in triangle.columns]
                triangle = triangle[premium_cols]
                
                # Add Premium
                if 'Premium' not in triangle.columns:
                    triangle['Premium'] = 1500000
                
                # Reorder
                cols = ['Accident_Year', 'Premium'] + [c for c in triangle.columns if c not in ['Accident_Year', 'Premium']]
                triangle = triangle[cols]
                
                # Rename to Lag_X format
                new_cols = ['Accident_Year', 'Premium']
                for c in triangle.columns[2:]:
                    try:
                        new_cols.append(f'Lag_{int(c)}')
                    except:
                        new_cols.append(str(c))
                triangle.columns = new_cols
                
                st.session_state.triangle = triangle
                st.success(f"✓ Loaded {len(triangle)} years!")
                
                st.subheader("Claims Triangle (Cumulative)")
                st.dataframe(triangle, width='stretch')
            else:
                st.error(f"Missing columns")
                
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
    
    lags = [0, 3, 6, 12, 24, 36]
    
    if 'triangle' not in st.session_state:
        data = []
        for i, ay in enumerate(ays):
            row = [ay, prems[i] if i < len(prems) else 1500000]
            row += [0] * len(lags)
            data.append(row)
        
        cols = ['Accident_Year', 'Premium'] + [f'Lag_{l}' for l in lags]
        st.session_state.triangle = pd.DataFrame(data, columns=cols)
    
    triangle_df = st.data_editor(
        st.session_state.triangle,
        column_config={
            "Accident_Year": st.column_config.NumberColumn("Year", disabled=True),
            "Premium": st.column_config.NumberColumn("Premium", format="$"),
            **{f"Lag_{l}": st.column_config.NumberColumn(f"Lag {l}m", format="$") for l in lags}
        },
        width='stretch', hide_index=True
    )

# Calculations
def calc_cl(triangle):
    lag_cols = []
    for c in triangle.columns:
        if 'Lag_' in c:
            try:
                num = int(c.split('_')[1])
                lag_cols.append(num)
            except:
                pass
    
    lag_cols = sorted(lag_cols)
    
    if len(lag_cols) < 2:
        return {'reserves': [0]*len(triangle), 'ultimates': [0]*len(triangle), 'ldfs': {}, 'error': 'Need at least 2 lag periods'}
    
    # Calculate LDFs
    ldfs = {}
    for i in range(len(lag_cols)-1):
        vals = []
        for idx in range(len(triangle)):
            c_val = triangle.loc[idx, f'Lag_{lag_cols[i]}']
            n_val = triangle.loc[idx, f'Lag_{lag_cols[i+1]}']
            if c_val > 0 and n_val > 0:
                vals.append(n_val / c_val)
        
        if vals:
            ldfs[f'{lag_cols[i]}-{lag_cols[i+1]}'] = np.mean(vals)
    
    # Calculate ultimate
    tail = 1.03
    cum_factor = np.prod(list(ldfs.values())) * tail if ldfs else tail
    
    reserves = []
    ultimates = []
    for idx in range(len(triangle)):
        latest = triangle.loc[idx, f'Lag_{max(lag_cols)}']
        ult = latest * cum_factor
        reserves.append(ult - latest)
        ultimates.append(ult)
    
    return {'reserves': reserves, 'ultimates': ultimates, 'ldfs': ldfs}

def calc_bf(triangle, elr):
    pct = {12: 0.75, 24: 0.92, 36: 0.97}.get(valuation_lag, 0.98)
    reserves = []
    ultimates = []
    for idx in range(len(triangle)):
        prem = triangle.loc[idx, 'Premium'] if 'Premium' in triangle.columns else 0
        if prem <= 0:
            prem = 1500000
        ult = prem * elr
        reserves.append(ult * (1 - pct))
        ultimates.append(ult)
    return {'reserves': reserves, 'ultimates': ultimates}

# Run
if st.button("🚀 RUN RESERVING MODEL", type="primary", width='stretch'):
    if triangle_df is None or triangle_df.empty:
        st.error("No data!")
    else:
        cl = calc_cl(triangle_df)
        bf = calc_bf(triangle_df, expected_lr)
        
        if 'error' in cl:
            st.error(cl['error'])
        else:
            blend_r = [(cl['reserves'][i] + bf['reserves'][i])/2 for i in range(len(triangle_df))]
            blend_u = [(cl['ultimates'][i] + bf['ultimates'][i])/2 for i in range(len(triangle_df))]
            
            results = pd.DataFrame({
                'Accident_Year': triangle_df['Accident_Year'],
                'Premium': triangle_df['Premium'],
                'Latest_Paid': [triangle_df.loc[i, f'Lag_{max([int(c.split("_")[1]) for c in triangle_df.columns if "Lag_" in c])}'] for i in range(len(triangle_df))],
                'CL_Reserve': cl['reserves'],
                'CL_Ultimate': cl['ultimates'],
                'BF_Reserve': bf['reserves'],
                'BF_Ultimate': bf['ultimates'],
                '50_50_Reserve': blend_r,
                '50_50_Ultimate': blend_u
            })
            
            st.subheader("Results by Accident Year")
            st.dataframe(results, width='stretch')
            
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CL Total", f"${sum(cl['reserves']):,.0f}")
            c2.metric("BF Total", f"${sum(bf['reserves']):,.0f}")
            c3.metric("50/50 Total", f"${sum(blend_r):,.0f}")
            c4.metric("Ultimate Total", f"${sum(blend_u):,.0f}")
            
            if cl['ldfs']:
                st.subheader("Development Factors")
                ldf_df = pd.DataFrame(list(cl['ldfs'].items()), columns=['Period', 'Factor'])
                st.dataframe(ldf_df, width='stretch')
            
            st.download_button("Download CSV", results.to_csv(index=False), "reserves.csv")