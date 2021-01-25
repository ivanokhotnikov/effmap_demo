import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import BytesIO
from joblib import dump, load
from effmap.hst import HST
from effmap.regressor import Regressor
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, train_test_split, cross_validate


@st.cache
def plot_validation():
    """Performs validation of the efficiency model by computing the volumetric efficiency of the benchmark HST, for which test data is available containing results of measurements of the input/output speeds, and comparing the computed efficiency with the test data.

    Returns:
    ---
        fig: plotly figure object
    """
    data = pd.read_csv(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/data/test_data.csv'
    )
    data.dropna(
        subset=['Forward Speed', 'Reverse Speed', 'Volumetric at 1780RPM'],
        inplace=True)
    speeds = data[['Forward Speed', 'Reverse Speed']].astype(float)
    speeds = speeds.stack()
    vol_eff = speeds / 1780 * 1e2
    piston_max = 1.1653 * 25.4 * 1e-3
    piston_min = 1.1650 * 25.4 * 1e-3
    bore_max = 1.1677 * 25.4 * 1e-3
    bore_min = 1.1671 * 25.4 * 1e-3
    rad_clearance_max = (bore_max - piston_min) / 2
    rad_clearance_min = (bore_min - piston_max) / 2
    benchmark = HST(196, swash=15, oil='SAE 30', oil_temp=60)
    benchmark.compute_sizes(k1=.7155, k2=.9017, k3=.47, k4=.9348, k5=.9068)
    eff_min = benchmark.compute_eff(speed_pump=1780,
                                    pressure_discharge=207,
                                    pressure_charge=14,
                                    h3=rad_clearance_max)
    eff_max = benchmark.compute_eff(speed_pump=1780,
                                    pressure_discharge=207,
                                    pressure_charge=14,
                                    h3=rad_clearance_min)
    fig = ff.create_distplot([vol_eff], ['Test data'],
                             show_hist=True,
                             bin_size=.3,
                             show_rug=False)
    fig.add_scatter(
        x=[eff_max['hst']['volumetric'], eff_max['hst']['volumetric']],
        y=[0, .6],
        mode='lines',
        name='Prediction. Min clearance',
        line=dict(width=1.5, ),
    )
    fig.add_scatter(
        x=[eff_min['hst']['volumetric'], eff_min['hst']['volumetric']],
        y=[0, .6],
        mode='lines',
        name='Prediction. Max clearance',
        line=dict(width=1.5, ),
    )
    fig.add_scatter(
        x=[vol_eff.mean(), vol_eff.mean()],
        y=[0, .6],
        mode='lines',
        name='Test mean',
        line=dict(width=1.5, dash='dash'),
    )
    fig.add_scatter(
        x=[vol_eff.mean() + vol_eff.std(),
           vol_eff.mean() + vol_eff.std()],
        y=[0, .6],
        mode='lines',
        name='Test mean + STD',
        line=dict(width=1.5, dash='dash'),
    )
    fig.add_scatter(
        x=[vol_eff.mean() - vol_eff.std(),
           vol_eff.mean() - vol_eff.std()],
        y=[0, .6],
        mode='lines',
        name='Test mean - STD',
        line=dict(width=1.5, dash='dash'),
    )
    fig.update_layout(
        title=
        f'Sample of {len(vol_eff)} measurements of the {benchmark.displ} cc/rev HST with {benchmark.oil} at {benchmark.oil_temp}C',
        width=800,
        height=500,
        xaxis=dict(
            title='HST volumetric efficiency, %',
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[84, 94],
            dtick=2,
        ),
        yaxis=dict(
            title='Probability density',
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[0, .6],
        ),
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,0)',
        showlegend=True,
        legend_orientation='h',
        legend=dict(x=0, y=-.15))
    return fig


def fit_catalogues(data_in):
    """
    Fits the custom Regressor model to the catalogue data. Each modelis cross_validated with the metrics being saved in the model's properties.

    Parameters:
    ---
    data_in: pd.DataFrame
        Catalogue data

    Returns:
    ---
    models: dict
        Dictionary of the fitted models. Keys: 'pump_mass', 'pump_speed', 'motor_mass', 'motor_speed'. Values: Regressor objects
    """
    models = {}
    if not os.path.exists('models'):
        os.mkdir('models')
    for machine_type in ('pump', 'motor'):
        for data_type in ('speed', 'mass'):
            data = data_in[data_in['type'] == f'{machine_type.capitalize()}']
            model = Regressor(machine_type=machine_type, data_type=data_type)
            x_full = data['displacement'].to_numpy(dtype='float64')
            y_full = data[data_type].to_numpy(dtype='float64')
            x_train, x_test, y_train, y_test = train_test_split(x_full,
                                                                y_full,
                                                                test_size=0.2,
                                                                random_state=0)
            strat_k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(
                model,
                x_train,
                y_train,
                cv=strat_k_fold,
                scoring=['neg_root_mean_squared_error', 'r2'],
                return_estimator=True,
                n_jobs=-1,
                verbose=0)
            model.r2_ = np.mean([k for k in cv_results['test_r2']])
            model.cv_rmse_ = -np.mean(
                [k for k in cv_results['test_neg_root_mean_squared_error']])
            model.cv_r2std_ = np.std([k for k in cv_results['test_r2']])
            model.cv_rmsestd_ = np.std(
                [k for k in cv_results['test_neg_root_mean_squared_error']])
            model.coefs_ = np.mean([k.coefs_ for k in cv_results['estimator']],
                                   axis=0)
            model.test_rmse_, model.test_r2_ = model.eval(x_test, y_test)
            model.fitted_ = True
            dump(model,
                 os.path.join('models', f'{machine_type}_{data_type}.joblib'))
            models['_'.join((machine_type, data_type))] = model
    return models


@st.cache
def plot_catalogues(models, data_in):
    """
    Plots the catalogue data with regression models

    Parameters:
    ---
    models: dict
        Dictionary of regression models. Keys: 'pump_mass', 'pump_speed', 'motor_mass', 'motor_speed'. Values: Regressor objects
    data_in: pd.DataFrame
        Catalogue data

    Returns:
    ---
    fig: plotly figure object
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.07,
    )
    for i, j in enumerate(models):
        model = models[j]
        data_type = model.data_type
        machine_type = model.machine_type
        data = data_in[data_in['type'] == f'{machine_type.capitalize()}']
        x = data['displacement'].to_numpy(dtype='float64')
        x_cont = np.linspace(.2 * np.amin(x), 1.2 * np.amax(x), num=100)
        for l in zip(('Regression model', 'Upper limit', 'Lower limit'),
                     (0, model.test_rmse_, -model.test_rmse_)):
            fig.add_scatter(x=x_cont,
                            y=model.predict(x_cont) + l[1],
                            mode='lines',
                            name=l[0],
                            line=dict(width=1, dash='dash'),
                            row=1 - (-1)**(i) // 2,
                            col=(i + 2) // 2)
        for idx, k in enumerate(data['manufacturer'].unique()):
            fig.add_scatter(x=data['displacement'][data['manufacturer'] == k],
                            y=data[data_type][data['manufacturer'] == k],
                            mode='markers',
                            name=k,
                            marker_symbol=idx,
                            marker=dict(size=7,
                                        line=dict(color='black', width=.5)),
                            row=1 - (-1)**(i) // 2,
                            col=(i + 2) // 2)
        fig.update_xaxes(
            title_text=f'{machine_type.capitalize()} displacement, cc/rev',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[0, round(1.1 * max(data['displacement']), -2)],
            row=1 - (-1)**(i) // 2,
            col=(i + 2) // 2)
        fig.update_yaxes(
            title_text=f'{data_type.capitalize()}, rpm'
            if data_type == 'speed' else f'{data_type.capitalize()}, kg',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[0, round(1.2 * max(data[data_type]), -2)]
            if data_type == 'mass' else [
                round(.7 * min(data[data_type]), -2),
                round(1.2 * max(data[data_type]), -2)
            ],
            row=1 - (-1)**(i) // 2,
            col=(i + 2) // 2)
    fig.update_layout(
        width=800,
        height=800,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,0)',
        showlegend=False,
    )
    return fig


def set_defaults(analysis_type):
    """Assigns the default values of oil, its parameters and initial design parameters to initialize a HST to plot its efficiency map and conduct calculations.

    Paramters:
    ---
    analysis_type: str, 'map' or 'comparison
        A string flag defining a type of chart to customize.

    Returns:
    ---
    'map':
        oil: string, default 'SAE 15W40'
        oil_temp: int, default 100
        max_displ: int, default 440
        max_power: int, default 685
        gear_ratio: float, default .75
        max_speed: int, default 2400
        max_pressure: int, default 650

    'comparison':
        displ_1: int, default 440
        displ_2: int, default 330
        speed: int, default 2025
        pressure: int, default 475
    """
    if analysis_type == 'sizing':
        displ = st.slider('Displacement, cc/rev', 100, 800, 440, 5)
        max_swash_angle = st.slider('Max swash angle, deg', 15, 21, 18, 1)
        pistons = st.slider('Number of pistons', 5, 13, 9, 2)
        return displ, max_swash_angle, pistons
    if analysis_type == 'performance':
        input_speed = st.slider('HST input (pump) speed, rpm', 1000, 3500,
                                2025, 5)
        pressure_charge = st.slider('Charge pressure, bar', 5, 70, 25, 5)
        pressure_discharge = st.slider('Discharge pressure, bar', 100, 1000,
                                       475, 5)
        return input_speed, pressure_charge, pressure_discharge
    if analysis_type == 'map':
        max_speed = st.slider('Max plotted speed, rpm', 1000, 4000, 2400, 100)
        max_pressure = st.slider('Max plotted pressure, bar', 100, 1000, 650,
                                 50)
        max_power = st.slider('Max absorbed power, kW', 200, 1100, 685, 5)
        gear_ratio = st.slider('Input gear ratio', .5, 2., .75, .05)
        return max_speed, max_pressure, max_power, gear_ratio
    if analysis_type == 'comparison':
        st.header('Comparison chart')
        displ_1 = st.slider('Displacement 1, cc/rev', 100, 800, 440, 5)
        displ_2 = st.slider('Displacement 2, cc/rev', 100, 800, 330, 5)
        speed = st.slider('Input speed, rpm', 1000, 3500, 2025, 5)
        pressure = st.slider('Discharge pressure, bar', 200, 800, 475, 5)
        oil_temp = st.slider('Temperature, C', 0, 100, 100, 10)
        return displ_1, displ_2, speed, pressure, oil_temp


@st.cache
def process_catalogues(mode='app'):
    """
    Creates and returns the regression models as well as the catalogue data. Depending on the flag value, the models are either loaded from the github repo, or being fit to the catalogue data.

    Parameters:
    ---
    mode: str, default 'app'
        Flag to between 'train' and 'app' modes. If 'app' is selected, the models are being read from github repo. If 'train' - the fit_catalogues(data) is called and the models are build from scratch.

    Returns:
    ---
    models: dict
        Dictionary of four regression models with te following keys: 'pump_mass', 'pump_speed', 'motor_mass', 'motor_speed'
    data: pd.DataFrame
        Catalogues data
    """
    data = pd.read_csv(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/data/data.csv',
        index_col='#')
    models = {}
    if mode == 'train':
        if os.path.exists('.\\models') and len(os.listdir('.\\models')):
            for file in os.listdir('.\\models'):
                models[file[:-7]] = load(
                    os.path.join(os.getcwd(), 'models', file))
        else:
            models = fit_catalogues(data)
    elif mode == 'app':
        for machine_type in ('pump', 'motor'):
            for data_type in ('mass', 'speed'):
                link = f'https://github.com/ivanokhotnikov/effmap_demo/blob/master/models/{machine_type}_{data_type}.joblib?raw=true'
                mfile = BytesIO(requests.get(link).content)
                models['_'.join((machine_type, data_type))] = load(mfile)
    return models, data


@st.cache
def plot_hsu(hst, models, max_speed, pressure_charge, max_pressure):
    """
    For the given HST computes the sizes, efficiencies and plots the efficiency map.

    Returns:
    ---
    fig: plotly figure object
    """
    hst.compute_sizes()
    hst.compute_speed_limit(models['pump_speed'])
    hst.add_no_load((1800, 140), (2025, 180))
    return hst.plot_eff_maps(max_speed,
                             max_pressure,
                             pressure_charge=pressure_charge,
                             show_figure=False,
                             save_figure=False)


@st.cache
def plot_comparison(displ_1, displ_2, speed, pressure, temp):
    """
    Prints a bar plot to compare total efficiencies of two HSTs.

    Parameters:
    ---
    displ_1, displ_2: float
        Displacements of the HSTs to be comapred
    speed, pressure, temp, charge: floats
        Operational parameters for the comparison

    Returns:
    ---
    fig: plotly figure object
    """
    effs_1, effs_2 = [], []
    motor_pows_1, motor_pows_2 = [], []
    pump_pows_1, pump_pows_2 = [], []
    oils = ('SAE 15W40', 'SAE 10W40', 'SAE 10W60', 'SAE 5W40', 'SAE 0W30',
            'SAE 30')
    hst_1, hst_2 = HST(displ_1, oil_temp=temp), HST(displ_2, oil_temp=temp)
    hst_1.compute_sizes()
    hst_2.compute_sizes()
    for oil in oils:
        hst_1.oil, hst_2.oil = oil, oil
        hst_1.load_oil()
        hst_2.load_oil()
        eff_1 = hst_1.compute_eff(speed, pressure)
        eff_2 = hst_2.compute_eff(speed, pressure)
        effs_1.append(eff_1['hst']['total'])
        effs_2.append(eff_2['hst']['total'])
        motor_pows_1.append(hst_1.performance['motor']['power'])
        motor_pows_2.append(hst_2.performance['motor']['power'])
        pump_pows_1.append(hst_1.performance['pump']['power'])
        pump_pows_2.append(hst_2.performance['pump']['power'])
    fig_eff = go.Figure()
    fig_eff.add_trace(
        go.Bar(
            x=oils,
            y=effs_1,
            text=[f'{eff:.2f}' for eff in effs_1],
            textposition='auto',
            name=f'{displ_1} cc/rev',
            marker_color='steelblue',
        ))
    fig_eff.add_trace(
        go.Bar(x=oils,
               y=effs_2,
               text=[f'{eff:.2f}' for eff in effs_2],
               textposition='auto',
               name=f'{displ_2} cc/rev',
               marker_color='indianred'))
    fig_eff.update_layout(
        title=
        f'Total efficiency of {displ_1} and {displ_2} cc/rev HSTs at {speed} rpm, {pressure} bar, {temp}C oil',
        yaxis=dict(
            title='Total HST efficiency, %',
            range=[50, 90],
        ),
        template='none',
        showlegend=True,
        legend_orientation='h')
    fig_pow = go.Figure()
    fig_pow.add_trace(
        go.Bar(
            x=oils,
            y=pump_pows_1,
            text=[f'{pow:.2f}' for pow in pump_pows_1],
            textposition='auto',
            name=f'{displ_1} cc/rev in',
            marker_color='steelblue',
        ))
    fig_pow.add_trace(
        go.Bar(
            x=oils,
            y=motor_pows_1,
            text=[f'{pow:.2f}' for pow in motor_pows_1],
            textposition='auto',
            name=f'{displ_1} cc/rev  out',
            marker_color='lightblue',
        ))
    fig_pow.add_trace(
        go.Bar(
            x=oils,
            y=pump_pows_2,
            text=[f'{pow:.2f}' for pow in pump_pows_2],
            textposition='auto',
            name=f'{displ_2} cc/rev  in',
            marker_color='indianred',
        ))
    fig_pow.add_trace(
        go.Bar(x=oils,
               y=motor_pows_2,
               text=[f'{pow:.2f}' for pow in motor_pows_2],
               textposition='auto',
               name=f'{displ_2} cc/rev out',
               marker_color='pink'))
    fig_pow.update_layout(
        title=
        f'Power balance of {displ_1} and {displ_2} cc/rev HSTs at {speed} rpm, {pressure} bar, {temp}C oil',
        yaxis=dict(title='Power, kW', ),
        template='none',
        showlegend=True,
        legend_orientation='h')
    return fig_eff, fig_pow


def main(mode):
    """
    Creates the dashboard, visualizes perfroamnce of the HST. Runs through the functionality of the Regressor and HST classes.
    """
    st.title('Design and Performance of a Hydrostatic Transmission (HST)')
    st.header('Pump sizing')
    st.subheader('General design of an axial piston pump')
    st.image(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/APM.png',
        use_column_width=True)
    st.write(
        '1 - pump shaft, 2 - pump outer bearing, 3 - swash plate, 4 - piston shoe/slipper, 5 - piston, 6 - cylinder block, 7 - valve plate, 8 - manifold/distribution block'
    )
    st.subheader('Sizing formulas')
    st.write('Piston diameter')
    st.latex(r'd = \sqrt[3]{\frac{4 k_1 V}{z^2 \tan{\gamma}}}')
    st.write('Pitch-circle diameter (PCD)')
    st.latex(r'D = \frac{zd}{\pi k_1}')
    st.write('Piston stroke')
    st.latex(r'h = D \tan \gamma')
    st.markdown(
        r'$V$ - pump displacement, $z$ - number of pistons, $\gamma$ - max swash angl, $k_1$ - PCD balance'
    )
    st.subheader('Setting pump parameters')
    hst = HST(*set_defaults('sizing'))
    hst.compute_sizes()
    st.subheader('Key sizes')
    st.write('All in millimeters, nominals')
    st.dataframe({
        'size':
        pd.Series(
            [hst.sizes['d'] * 1e3, hst.sizes['D'] * 1e3, hst.sizes['h'] * 1e3],
            index=[
                'Piston diameter', 'Pitch-circle diameter', 'Piston stroke'
            ])
    })
    models, data = process_catalogues(mode)
    hst.compute_speed_limit(models['pump_speed'])
    st.subheader('Pump speed limits (see Catalogue data and Regressions)')
    st.write('Speed in rpm')
    st.dataframe({
        'speed':
        pd.Series(hst.pump_speed_limit,
                  index=['Min rated speed', 'Rated speed', 'Max rated speed'])
    })
    st.header('HST performance')
    st.subheader('Efficiencies formulas')
    st.write('Pump volumetric efficiency')
    st.latex(r'\eta_{Pv} = 1 - \frac{\Delta p}{\beta} - \frac{Q_L}{Q_{Pth}}')
    st.latex(r'Q_{Pth} = n_P V_P')
    st.write('Motor volumetric efficiency')
    st.latex(r'\eta_{Mv} = 1 - \frac{Q_L}{Q_{Mth}}')
    st.latex(r'Q_{Mth} = n_M V_M')
    st.write('HST volumetric efficiency')
    st.latex(r'\eta_{HSTv} = \eta_{Pv} \eta_{Mv}')
    st.markdown(
        r"$\Delta p = p_{h} - p_{l}$ - difference between the high (or discharge) $p_h$ and low (or charge) $p_l$ pressure levels of an HST's lines, $\beta$ - oil bulk modulus, $Q_L$ - total machine's leakage volume flow rate, $Q_{Pth}$, $Q_{Mth}$ - pump's and motor's theoretical volume flow rates, $n_P, n_M$ - pump's and motor's shaft speeds, $V_P, V_M$ - pump's and motor's displacements"
    )
    st.write('Pump mechanical efficiency')
    st.latex(
        r'	\eta_{Pm} = 1 - A_P \exp\left(-\frac{\mu n_P B_{P}}{\Delta p\gamma_P}\right) - C_P\sqrt{\frac{\mu n_P}{{\Delta p}\gamma_P }}-\frac{D_P}{\Delta p \gamma_P}'
    )
    st.write('Motor mechanical efficiency')
    st.latex(
        r'\eta_{Mm} = 1 - A_M \exp\left(-\frac{\mu n_M B_{M}}{\Delta p\gamma_M}\right) - C_M \sqrt{\frac{\mu n_M}{{\Delta p}\gamma_M }}-\frac{D_M}{\Delta p\gamma_M}'
    )
    st.write('HST mechanical efficiency')
    st.latex(r'\eta_{HSTm} = \eta_{Pm} \eta_{Mm}')
    st.markdown(
        r"$A_P, B_P, C_P, D_P, A_M, B_M, C_M, D_M$ - empirical coefficients of pump's and motor's mechanical efficiencies, $\mu$ - oil dynamic viscosity, $\gamma_P, \gamma_M$ - pump and motor swash angles"
    )
    st.write('Pump total efficiency')
    st.latex(r'\eta_{P} = \eta_{Pv} \eta_{Pm}')
    st.write('Motor total efficiency')
    st.latex(r'\eta_{M} = \eta_{Mv} \eta_{Mm}')
    st.write('HST total efficiency')
    st.latex(r'\eta_{HST} = \eta_{P} \eta_{M}')
    st.latex(r'\eta_{HST} = \eta_{HSTv} \eta_{HSTm}')
    st.subheader('Performance formulas')
    st.write('Averaged actual volume flow rate in the HST')
    st.latex(
        r'Q_{act} = \eta_{Pv} Q_{Pth} = \left(1 - \frac{\Delta p}{\beta} \right) Q_{Pth} - Q_{L}'
    )
    st.write('Pump torque')
    st.latex(r'	T_P=\frac{\Delta p V_P}{\eta_{Pm}}')
    st.write('Motor speed')
    st.latex(
        r'n_M = \frac{\eta_{Mv} Q_{act}}{V_M} = \eta_{HSTv} \frac{V_P}{V_M} n_P'
    )
    st.write('Motor torque')
    st.latex(
        r'T_{M} = \eta_{Mm} \Delta p V_M = \eta_{HSTm} \frac{V_M}{V_P} T_P')
    st.write('Absorbed power (pump shaft power)')
    st.latex(r'P_{in} = n_P T_P')
    st.write('Transmitted power (motor shaft power)')
    st.latex(r'P_{out} = n_M T_M = \eta_{HST} P_{in}')
    st.subheader('Loading scheme of an axial piston pump')
    st.image(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/APM_1.png',
        use_column_width=True)
    st.subheader('Averaged structural loads formulas')
    st.write('Shaft radial load')
    st.latex(
        r'\overline{F}_r = \left( \left\lceil \frac{z}{2} \right\rceil p_h + \left\lfloor \frac{z}{2} \right\rfloor p_l  \right) A_p \tan \gamma'
    )
    st.write('Swash plate high-pressure side longitudinal load (X)')
    st.latex(
        r'\overline{F}_{sw.hp.X} = \left\lceil \frac{z}{2} \right\rceil p_h A_p'
    )
    st.write('Swash plate high-pressure side transversal load (Z)')
    st.latex(
        r'\overline{F}_{sw.hp.Z} = \left\lceil \frac{z}{2} \right\rceil p_h A_p \tan \gamma'
    )
    st.write('Swash plate low-pressure side longitudinal load (X)')
    st.latex(
        r'\overline{F}_{sw.lp.X} = \left\lfloor \frac{z}{2} \right\rfloor p_l A_p'
    )
    st.write('Swash plate low-pressure side transversal load (Z)')
    st.latex(
        r'\overline{F}_{sw.lp.Z} = \left\lfloor \frac{z}{2} \right\rfloor p_l A_p \tan \gamma'
    )
    st.write('Motor high-pressure side normal load')
    st.latex(
        r'\overline{F}_{m.hp} = \left\lceil\frac{z}{2}\right\rceil p_h A_p\frac{1}{\cos \gamma}'
    )
    st.write('Motor low-pressure side normal load')
    st.latex(
        r'\overline{F}_{m.lp} = \left\lfloor\frac{z}{2}\right\rfloor p_l A_p\frac{1}{\cos \gamma}'
    )
    st.subheader('Parameters')
    hst.oil = st.selectbox('Oil', ('SAE 15W40', 'SAE 10W40', 'SAE 10W60',
                                   'SAE 5W40', 'SAE 0W30', 'SAE 30'))
    hst.load_oil()
    st.subheader('Physical properties of oil')
    st.write(hst.plot_oil())
    hst.oil_temp = st.slider('Oil temperature, C', 0, 100, 100, 10)
    input_speed, pressure_charge, pressure_discharge = set_defaults(
        'performance')
    st.subheader('Efficiencies')
    st.write(
        'Note: pump and motor have same design, displacement, max swash angle and number of pistons. Hydraulic lines between the pump and the motor do not introduce power losses'
    )
    st.write('Percentage')
    st.dataframe(
        hst.compute_eff(input_speed,
                        pressure_discharge,
                        pressure_charge=pressure_charge))
    st.subheader('Performance')
    st.write('Speed in rpm, torque in Nm, power in kW')
    st.write(pd.DataFrame(hst.performance)[['pump', 'motor', 'delta']])
    st.subheader('Resultant structural loads')
    st.write('Force in kN, torque in Nm, pressure in bar')
    hst.compute_loads(pressure_discharge)
    st.write(
        pd.DataFrame({
            'load':
            pd.Series([
                pressure_discharge, pressure_charge, hst.shaft_radial,
                hst.shaft_torque, hst.swash_hp_x, hst.swash_hp_z,
                hst.swash_lp_x, hst.swash_lp_z, hst.motor_hp, hst.motor_lp
            ],
                      index=[
                          'Discharge pressure', 'Charge pressure',
                          'Shaft radial', 'Shaft torque', 'Swash plate HP (X)',
                          'Swash plate HP (Z)', 'Swash plate LP (X)',
                          'Swash plate LP (Z)', 'Motor HP (Normal)',
                          'Motor LP (Normal)'
                      ])
        }))
    st.header('Efficiency map')
    st.subheader('Parameters')
    max_speed, max_pressure, hst.max_power_input, hst.input_gear_ratio = set_defaults(
        'map')
    temp_engine = st.selectbox(
        'Engine', ('Engine 1', 'Engine 2', 'Engine 3', 'Engine 4'))
    hst.engine = temp_engine.lower().replace(" ", "_")
    st.write(plot_hsu(hst, models, max_speed, pressure_charge, max_pressure))
    st.header('Comparison of HSTs\' sizes and oils')
    fig_eff, fig_pow = plot_comparison(*set_defaults('comparison'))
    st.write(fig_eff)
    st.write(fig_pow)
    st.header('Validation of the HST efficiency model')
    st.write(plot_validation())
    st.header('Catalogue data and regressions')
    st.write(plot_catalogues(models, data))
    for i in models:
        units = 'rpm' if 'speed' in i else 'kg'
        st.write(
            f'RMSE {models[i].machine_type} {models[i].data_type} = {np.round(models[i].test_rmse_, decimals=2)}',
            u'\u00B1', f'{np.round(models[i].cv_rmsestd_,decimals=2)}', units)


if __name__ == '__main__':
    main(mode='app')
