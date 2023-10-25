import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import fun_static_model as fsm

st.set_page_config(layout="wide")

st.title("Musculoskeletal modelling")

st.write("The aim of this web app is to calculate and compare the function of the feeding system based on measurments made on 3D virtual models.")

st.header('Jaw muscle morphology of primates')
st.markdown("Data on the morphology of the species were collected during quantitative dissections. The **head width** and **lower jaw length** (in mm) and **muscle physiological cross section areas** (PCSA, cm<sup>2</sup>) for the three muscle groups in each species are listed in the table below.", unsafe_allow_html=True)

data_morpho = pd.DataFrame({'species':['m_fascicularis', 'm_murinus'], 'head_width':[58, 22], 'lower_jaw_length':[53, 22], 'pcsa_masseter':[2.417, 0.664*1.2], 'pcsa_pterygoid_medial':[1.11, 0.146*1.2], 'pcsa_temporalis':[3.596*1.1, 0.623*1.2], 'pcsa_total':[7.123, 1.433*1.2]})
st.table(data_morpho)

st.header('Load your measurments')
st.write("First upload the table containing the muscule coordinates and PCSA, and then the table containing the coordinates of the jaw joint and bite point.")
muscle_file = st.file_uploader("Table containing the muscle data", accept_multiple_files=False)
geom_file = st.file_uploader("Table containing the joint data", accept_multiple_files=False)

column1, column2 = st.columns(2)

if (muscle_file and geom_file) is not None:
    
    df_muscle = pd.read_csv(muscle_file)
    df_geom = pd.read_csv(geom_file)

    grouped_pcsa = data_morpho.groupby(['species'])
    grouped_muscle = df_muscle.groupby(['species'])
    grouped_geom = df_geom.groupby(['species'])

    # add the pcsa to the coordinate table
    
    list_df_muscle_pcsa = []
    for i in grouped_muscle.groups:
        df_muscle_pcsa_i = fsm.fun_insert_pcsa(grouped_muscle.get_group(i), grouped_pcsa.get_group(i))
        list_df_muscle_pcsa.append(df_muscle_pcsa_i)
        
    df_muscle_pcsa = pd.concat(list_df_muscle_pcsa)

    st.header('Overview of the uploaded data')
    st.write("You can assess in each tab the measurments you uploaded. For each muscle, the PCSA was divided by the number of strands representing the line of action and added as the last column in Muscle Table.")
    tab1, tab2 = st.tabs(['Muscle Tabe', 'Point Table'])
    with tab1:
        st.write(df_muscle_pcsa)   
    with tab2:
        st.write(df_geom)

    # static model: initial conditions
    
    st.header('Static model parameters')
    st.write("Once you have checked that your data tables are fine, you can define the initial conditions and parameters of your model.")
    measur_sides = st.selectbox('**Number of sides** on the cranium where the measurments have been made', (1, 2))
    gape_max = st.number_input('**Maximum gape angle (degrees)**: the model calculates the static equilibrium for each degree from 0 to the maximum gape angle.', value=0, min_value=0, step=1)
    f_fibre_max = st.number_input('**Maximum muscle fibre strength (N/cm^2)**: how much force an individual muscle fibre can generate.', value=25, min_value=18, step=1)

    grouped_df_muscle_pcsa = df_muscle_pcsa.groupby(['species'])
    
    list_freac = []
    list_muscle_force = []
    list_muscle_resultant = []
    list_muscle_strain = []
    list_muscle_mom = []
    list_muscle_mom_arm = []
    
    for i in grouped_df_muscle_pcsa.groups:
       freac, muscle_force, muscle_resultant, muscle_strain, muscle_mom, muscle_mom_arm = fsm.bite_model(grouped_df_muscle_pcsa.get_group(i), grouped_geom.get_group(i), gape_max, f_fibre_max, measur_sides)
       list_freac.append(freac)
       list_muscle_force.append(muscle_force)
       list_muscle_resultant.append(muscle_resultant)
       list_muscle_strain.append(muscle_strain)
       list_muscle_mom.append(muscle_mom)
       list_muscle_mom_arm.append(muscle_mom_arm)

    df_freac = pd.concat(list_freac)
    df_muscle_force = pd.concat(list_muscle_force)
    df_muscle_resultant = pd.concat(list_muscle_resultant)
    df_muscle_strain = pd.concat(list_muscle_strain)
    df_muscle_mom = pd.concat(list_muscle_mom)
    df_muscle_mom_arm = pd.concat(list_muscle_mom_arm)

    # summary table of the static model
    
    df_bite_model_summary = fsm.LeverParamMaxBiteForce(df_freac, df_muscle_force)
    df_bite_model_summary_melt = df_bite_model_summary.melt(id_vars=['species', 'gape_angle', 'gape_h', 'variable'], 
                                                        value_vars=df_bite_model_summary.drop(['species', 'gape_angle', 'gape_h', 'variable'], axis=1).columns,
                                                        var_name='bite_point', value_name='value')
    st.write(df_bite_model_summary_melt)
    
    # plots absolute values
    
    plot_pcsa = px.bar(data_morpho, x='species', y='pcsa_total', color='species', labels={'species':'Species', 'pcsa_total':'PCSA (cm^2)'}, template = 'seaborn')
    plot_pcsa.update_layout(showlegend=False)

    plot_Biteforce = px.bar(np.round(df_bite_model_summary_melt[df_bite_model_summary_melt['variable'] == 'BfMag'], 2), 
                                  x='species', 
                                  y='value', 
                                  color='species', 
                                  labels={'species':'Species', 'value':'Bite force (N)'},
                            template = 'seaborn')
    plot_Biteforce.update_layout(showlegend=False, bargap=0.2, width = 400)
    #plot_Biteforce.update_layout(bargroupgap=0.2, width = 500)
    

    st.header('Results')
    st.write("We start examining the first results.\n 1) What can you say about the feeding function of each species based on the plots below?\n 2) Based on the *in vivo* measurments presented in [Chazeau et al. (2012) *J. Zoology*](http://www.anthonyherrel.fr/publications/Chazeau%20et%20al%202013%20J%20Zool.pdf) (hint: look at Figure 2), how would you evaluate the validity of the model of *M. murinus*?\n 3) Do you think that the measurments plotted below are comparable between the different species?")
    
    cc1, cc2 = st.columns([1,1])
    with cc1:
        st.plotly_chart(plot_pcsa, use_container_width=True)
    with cc2:
        st.plotly_chart(plot_Biteforce, use_container_width=True)

    answer_1 = st.text_area("Type your answer and validate with Crtl+Enter", max_chars = 500, key = "answer_1")
    st.write(f'You wrote {len(answer_1)} characters.')

    button_clicked = False
    if st.button('Next'):
        st.write("Comparing the muscle PCSA and bite force is a bit difficult here because of important differences in head dimensions between the two species. For instance, the head width of the macaque is more than twice that of the mouse lemur. We hence need to standardise our measurments to interpret the results.\n Here, we will choose the lower jaw length (in mm) to standardise the results. The biting efficiency was calculated by dividing the bite force by the total muscle force: this is equivalent to the mechanical advantage. \n 1) How has the standardisation changed the previous results?\n 2) One of the two species shows a greater scaled bite force, how do you think it is achieved?\n")

        # select a measurment to normalise the data
        
        option_head_measurment = data_morpho.columns[2]
  
        # # normalise data to head dimension

        df_morpho_norm = fsm.fun_normalise_pcsa_hw(data_morpho, option_head_measurment)
        df_morpho_norm['pcsa_total'] = np.round(df_morpho_norm['pcsa_total'], 2)
        
        df_freac_norm = fsm.fun_batch_normalise_with_hw(df_freac, data_morpho, option_head_measurment)
        
        df_muscle_force_norm = fsm.fun_batch_normalise_with_hw(df_muscle_force, data_morpho, option_head_measurment)

        df_muscle_mom_arm_norm = fsm.fun_batch_normalise_with_hw(df_muscle_mom_arm, data_morpho, option_head_measurment)
        
        df_bite_model_norm_summary = fsm.LeverParamMaxBiteForce(df_freac_norm, df_muscle_force_norm)
        df_bite_model_norm_summary_melt = df_bite_model_norm_summary.melt(id_vars=['species', 'gape_angle', 'gape_h', 'variable'], 
                                                        value_vars=df_bite_model_norm_summary.drop(['species', 'gape_angle', 'gape_h', 'variable'], axis=1).columns,
                                                        var_name='bite_point', value_name='value')


        #st.write(df_freac, df_muscle_mom, df_muscle_mom_arm, df_muscle_mom_arm_norm.groupby('species').first())

        # plot the normalised data 
    
        plot_pcsa_norm = px.bar(df_morpho_norm, x='species', y='pcsa_total', color='species', labels={'species':'Species', 'pcsa_total':'PCSA^0.5/Lower jaw length'}, template = 'seaborn')
        plot_pcsa_norm.update_layout(showlegend=False, bargap=0.2, width = 400)
    
        plot_Biteforce_norm = px.bar(np.round(df_bite_model_norm_summary_melt[df_bite_model_norm_summary_melt['variable'] == 'BfMag'], 2), 
                                     x='species',
                                     y='value',
                                     color='species',
                                     labels={'species':'Species', 'value':'Bite force/Lower jaw length'}, template = 'seaborn')
        plot_Biteforce_norm.update_layout(showlegend=False, bargap=0.2, width = 400)

        plot_BitingEff = px.bar(np.round(df_bite_model_summary_melt[df_bite_model_summary_melt['variable'] == 'BitingEff'], 2), 
                                x='species', 
                                y='value', 
                                color='species', 
                                labels={'species':'Species', 'value':'Biting efficiency'}, template = 'seaborn')
        plot_BitingEff.update_layout(showlegend=False, bargap=0.2, width = 400)
        plot_BitingEff.update_layout(bargroupgap=0.2, width = 400)

        plot_JrfNormBf = px.bar(np.round(df_bite_model_summary_melt[df_bite_model_summary_melt['variable'] == 'JrfNormBf'], 2),
                                x='species', 
                                y='value', 
                                color='species', 
                                labels={'species':'Species', 'value':'Joint reaction force/Bite force'}, template = 'seaborn')
        plot_JrfNormBf.update_layout(showlegend=False, bargap=0.2, width = 400)

        df_muscle_mom_arm_norm_sb = df_muscle_mom_arm_norm.drop_duplicates('species')
        plot_outLeverLengthNorm = px.bar(np.round(df_muscle_mom_arm_norm_sb, 2),
                                         x='species', 
                                         y='outlever_length', 
                                         color='species', 
                                         labels={'species':'Species', 'outlever_length':'Out-lever length/Lower jaw length'}, template = 'seaborn')
        plot_outLeverLengthNorm.update_layout(showlegend=False, bargap=0.2, width = 400)
        
        plot_muscleMomArm = px.bar(np.round(df_muscle_mom_arm_norm, 2),
                                   x='species', 
                                   y='total_muscle_moment_arm', 
                                   color='species', 
                                   labels={'species':'Species', 'total_muscle_moment_arm':'Muscle moment arm/Lower jaw length'}, template = 'seaborn')
        plot_muscleMomArm.update_layout(showlegend=False, bargap=0.2, width = 400)

        plot_scatter_mom_arm = px.scatter(df_muscle_mom_arm_norm, x = "gape_angle", y = 'total_muscle_moment_arm', color='species',
                                          labels={"gape_angle": "Gape angle (degrees)",
                                                  "total_muscle_moment_arm": "Muscle moment arm/Lower jaw length",
                                                  "species": "Species"}, template = 'seaborn')
        plot_scatter_mom_arm.update_layout(showlegend=False, bargap=0.2, width = 400)

        cc1, cc2 = st.columns([1,1])
        with cc1:
            st.plotly_chart(plot_pcsa_norm, use_container_width=True)
            st.plotly_chart(plot_BitingEff, use_container_width=True)
        with cc2:
            st.plotly_chart(plot_Biteforce_norm, use_container_width=True)
            st.plotly_chart(plot_JrfNormBf, use_container_width=True)

        answer_2 = st.text_area("Type your answer and validate with Crtl+Enter", max_chars = 500, key = "answer_2")
        st.write(f'You wrote {len(answer_2)} characters.')
    
        st.write("We have just seen that *M. fascicularis* has a greater biting efficiency than *M. murinus*. We can assess what is allowing for a more efficient muscle force transmission in this species.\n 1) What are the two measurements that we have plotted below?\n 2) What do these two plots tell us about adductor muscle force transmission?\n 3) What could be the limitations of the musculoskeletal models we have just made?")
        
        cc3, cc4 = st.columns(2)
        with cc3:
            st.plotly_chart(plot_outLeverLengthNorm, use_container_width=True)
        with cc4:
            st.plotly_chart(plot_scatter_mom_arm, use_container_width=True)

        answer_3 = st.text_area("Type your answer and validate with Crtl+Enter", max_chars = 500, key = "answer_3")
        st.write(f'You wrote {len(answer_3)} characters.')

