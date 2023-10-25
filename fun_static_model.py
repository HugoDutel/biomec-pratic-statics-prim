import pandas as pd
import numpy as np
from numpy import linalg as la

def fun_normalise_pcsa_hw(data_pcsa, head_measurment):
    # normalise pcsa to head with
    
    grouped_data_pcsa = data_pcsa.groupby(['species'])
    list_data_pcsa_norm = []
    for i in grouped_data_pcsa.groups:
        list_data_pcsa_i = grouped_data_pcsa.get_group(i).filter(regex='pcsa', axis=1)**0.5/grouped_data_pcsa.get_group(i)[head_measurment].iloc[0]
        list_data_pcsa_norm.append(list_data_pcsa_i)
    df_data_pcsa_norm = pd.concat(list_data_pcsa_norm)
    df_data_pcsa_norm = pd.concat([data_pcsa[['species','head_width','lower_jaw_length']], df_data_pcsa_norm], axis=1, join='inner')
    return df_data_pcsa_norm

def fun_batch_normalise_with_hw(df_results, df_morpho, head_measurment):
    # applies fun_normalise_with_hw to each species of the result df
    
    grouped_df_results = df_results.groupby(['species'])
    grouped_df_morpho = df_morpho.groupby(['species'])
    list_df_results_norm = []
    for i in grouped_df_results.groups:
        results_norm_i = fun_normalise_with_hw(grouped_df_results.get_group(i), grouped_df_morpho.get_group(i), head_measurment)
        list_df_results_norm.append(results_norm_i)
    df_results_normalised = pd.concat(list_df_results_norm)
    return df_results_normalised

def fun_normalise_with_hw(df_static_res, df_morpho, head_measurment):
    # normalise result tables to head width

    df_static_res_norm = df_static_res.drop(['species', 'gape_angle', 'gape_h'], axis=1)/df_morpho[head_measurment].iloc[0]
    df_static_res_norm = pd.concat([df_static_res[['species','gape_angle','gape_h']], df_static_res_norm], axis=1, join='inner')
    return df_static_res_norm

def fun_insert_pcsa(df_muscle, df_pcsa):
    # add the pcsa columns to the muscle coordinates
    
    df_muscle_sub_mass = df_muscle[df_muscle['strand'].str.contains('masseter')]
    df_muscle_sub_pterygoid = df_muscle[df_muscle['strand'].str.contains('pterygoid')]
    df_muscle_sub_temporalis = df_muscle[df_muscle['strand'].str.contains('temporalis')]

    pcsa_masseter_divided = df_pcsa.filter(regex='masseter', axis=1)/len(df_muscle_sub_mass)
    pcsa_pterygoid_divided = df_pcsa.filter(regex='pterygoid', axis=1)/len(df_muscle_sub_pterygoid)
    pcsa_temporalis_divided = df_pcsa.filter(regex='temporalis', axis=1)/len(df_muscle_sub_temporalis)

    df_muscle_sub_mass.insert(len(df_muscle_sub_mass.columns), 'pcsa', pcsa_masseter_divided.iloc[0,0])
    df_muscle_sub_pterygoid.insert(len(df_muscle_sub_pterygoid.columns), 'pcsa', pcsa_pterygoid_divided.iloc[0,0])
    df_muscle_sub_temporalis.insert(len(df_muscle_sub_temporalis.columns), 'pcsa', pcsa_temporalis_divided.iloc[0,0])

    df_muscle_pcsa = pd.concat([df_muscle_sub_mass, df_muscle_sub_pterygoid, df_muscle_sub_temporalis])

    return df_muscle_pcsa

def RotInsertion(phi, lmt0_ins):
    
    Rz = [np.array([[np.cos(i), -np.sin(i), 0], [np.sin(i), np.cos(i), 0], [0, 0, 1]]) for i in phi]
    lmt_ins = [np.matmul(i, lmt0_ins.T).T for i in Rz]
    lmt_ins = np.array(lmt_ins)
    return lmt_ins


def UnitVector(lmt0_or, lmt_ins):
    
    strand_l = la.norm(lmt0_or - lmt_ins, axis = 2) # along the z axis, alternative to [la.norm(lmt0_or - i, axis=1) for i in lmt_ins]
    strand_u = (lmt0_or - lmt_ins) / strand_l[:,:,None] # unit vector for each muscle strand
    strand_u = np.nan_to_num(strand_u)
    return strand_l, strand_u


def MtuLength(lmt0, lmt0_or, lmt0_ins, lmt_ins):
    
    strand_l0 = la.norm(lmt0_or - lmt0_ins, axis=1) # initial length of the muscle strand
    strand_l = la.norm(lmt0_or - lmt_ins, axis=2) # length of the muscle strand at a given gape
    lmt = lmt0 - strand_l0 + strand_l
    lmt_strain = (lmt-lmt0)/lmt0
    
    return lmt, lmt_strain

def LeverParamMaxBiteForce(DfReactionForces, DfMuscleForces):
    '''
    Select gape angle where the Bite force is maximal. 
    Returns the Bite force, JRF, Biting efficiency, and JRF/Bite force for each bite point at each bite point.

    Parameters
    ----------
    DfReactionForces : Dataframe containing the bite force and joint reaction forces.
    DfMuscleForces : Dataframe containing the muscle forces.

    Returns
    -------
    Dataframe.

    '''
    
    DfReactionForces = DfReactionForces.reset_index() # rest the index of the df
    DfMuscleForces = DfMuscleForces.reset_index() # rest the index of the df
    
    refcol = DfReactionForces.filter(regex='^(?=.*bp)(?:(?!jrf).)*$', axis=1).columns[0] # take the first column where bite force is stored if severals
    DfReactionForces = DfReactionForces.loc[DfReactionForces[refcol].abs().groupby(DfReactionForces['species']).idxmax()] # for each species, select the entire row for which the bite force magnitude is max
    
    index = DfReactionForces.index.values # store the row indexes
    
    DfMuscleForces = DfMuscleForces.loc[index,] # select the corresponding rows in the muscle force dataframe
    
    df_fbite = DfReactionForces.filter(regex='^(?=.*bp)(?:(?!jrf).)*$', axis=1) # select the columns that do not contain JRF string
    df_jrfmag = DfReactionForces.filter(regex='jrfmag', axis=1) # select the columns that contain JRFmag string
    
    dfs = [df_fbite, df_jrfmag]
    df_fbite, df_jrfmag = [dfi.abs() for dfi in dfs] # return the magnitude of the forces (absolute value)
    
    df_bitingeff = df_fbite.div(DfMuscleForces['total_fm'], axis = 0)
    df_jrftobf = df_jrfmag.div(df_fbite.values, axis=0)
 
    df_jrfmag.columns = df_fbite.columns # for the jrf dfs, put the bite point as colnames
    df_jrftobf.columns = df_fbite.columns    
    
    df_jrfmag.insert(0, "variable", ['JrfMag']*len(df_jrfmag)) # insert a new column and duplicate the entry according to the number of rows in the df_fbite
    df_jrftobf.insert(0, "variable", ['JrfNormBf']*len(df_jrftobf)) 
    df_fbite.insert(0, "variable", ['BfMag']*len(df_fbite))    
    df_bitingeff.insert(0, "variable", ['BitingEff']*len(df_bitingeff))
       
    dfs = [df_fbite, df_jrfmag, df_bitingeff, df_jrftobf]
    dfs = [pd.concat([DfReactionForces[['species','gape_angle','gape_h']], dfi], axis=1, join='inner') for dfi in dfs]
    dfs = pd.concat(dfs)
    
    dfs = dfs.reset_index(drop=True)
    
    dfs.columns = dfs.columns.str.replace('bp', '')
    
    return dfs


def bite_model(df_muscle, df_geom, gape_max, f_fibre_max, measur_sides, lm_opt = 1.0, fl_width = 0.3):
    """
    Static bite model with rigid tendon
    ----------
    Parameters
    ----------
    outputdir: Directory where the results will be saved. E.g. .\\output\\
    data_mus : Dataframe
        Dataframe containg the muscle properties and coordinates. 
        Columns: ['species', 'muscle', 'strand', 'lmt0', 'lt0', 'lm0', 'penn0', 'ox', 'oy', 'oz', 'ix', 'iy', 'iz'].
    data_geom : Dataframe
        Contains the geometrical properties, joint and bite point position.
        Columns: ['species', 'variable', 'x', 'y', 'z', 'x', 'y', 'z'].
    gape_range : List of 2
        Gape angles at which the calculations are made.
        [0,0] = analysis at 0 degree
        [0,-40] = analysis from 0 to -40 degree with one degree increment
    lmt_opt : Real number, optional
        Optimal muscle fibre length. The default is '1.0'.
    fl_width : Real number, optional
        Width of the force-length hyperbol from Thelen 2003. The default is '0.3'.

    """
    # set the gape angles
    gape_angle = np.arange(0, -gape_max-1, -1)
    
    # extract geometrical data
    mandjr = np.array(df_geom.loc[df_geom['point'].str.contains("joint"), ['x','y','z']]) # right tmj
    bp = np.array(df_geom.loc[df_geom['point'].str.contains("bp"), ['x','y','z']]) # bite point
    bp_ref = bp-mandjr
    
    # extract muscle data
    lmt0_or = np.array(df_muscle[['ox','oy','oz']].values-mandjr) # origin of the muscle strand
    lmt0_ins = np.array(df_muscle[['ix','iy','iz']].values-mandjr) # insertion of the muscle strand
    lmt0 = la.norm(lmt0_or - lmt0_ins, axis = 1) # MTU length at rest
    pcsa = np.array(df_muscle['pcsa'])[None,:] # invidiual muscle strand pcsa
    
    # calculate the coordinates of insertions for each gape
    phi = np.deg2rad(gape_angle)
    lmt_ins = RotInsertion(phi, lmt0_ins)
    
    # gape height at the tip of the jaw
    gape_distance = np.sin(phi)*bp_ref[0,0]
    
    # vector and length of each muscle strand   
    strand_l, strand_u = UnitVector(lmt0_or, lmt_ins)
    
    # MTU length and strain 
    lmt, lmt_strain = MtuLength(lmt0, lmt0_or, lmt0_ins, lmt_ins)

    # normalised MTU length
    mtu_norm = lmt/lmt

    # multiply pcsa if measurments on one side
    if measur_sides == 1:
        pcsa = pcsa*2
    
    # maximal muscle isometric force
    fmax = pcsa*f_fibre_max
    
    # invidiual muscle strand force
    fmt = fmax*mtu_norm
    
    # Force component for each muscle strand
    Fm_cp = fmt[:,:,None]*strand_u
    
    # Moment around the jaw joint
    uvector = [0, 0, 1]
    M = np.cross(lmt_ins, Fm_cp)*uvector
    M_sum = np.sum(M[:,:,2], axis=1)
    
    # bite force
    bf = -M_sum[:,None]/bp_ref[:,0]
    hvect = np.sin((np.pi/2)-phi) 
    bf_y = (bf.T*hvect).T
    
    #%% magnitude of the jrf for each bite location
    Fm_cp_sum = np.sum(Fm_cp, axis = 1)
    jrfx = -Fm_cp_sum[:,0]
    jrfy = -(Fm_cp_sum[:,1,None]+bf_y)
    jrfz = -Fm_cp_sum[:,2]
    
    def create_jrf_array(jrfy_sel):
        a = np.column_stack((jrfx[:, None], jrfy_sel[:, None], jrfz[:, None]))
        b = la.norm(a, axis=1)
        c = np.column_stack((a, b))
        return c
    
    jrfy_col_index = range(jrfy.shape[1])
    jrf = [create_jrf_array(jrfy[:, col_index]) for col_index in jrfy_col_index]
    
    # species
    species =  df_muscle['species'].unique()
    
    # gape distance #gape.insert(0, 'species', species)
    gape = pd.DataFrame({'species':species[0],'gape_angle':-gape_angle, 'gape_h':-gape_distance})
    
    # bite force and joint reaction force
    bf_y = pd.DataFrame(bf_y) 
    bf_y.columns = df_geom.loc[df_geom['point'].str.contains("bp"), 'point']    
    jrf = pd.DataFrame(np.column_stack(jrf))
    bp_ref = df_geom.loc[df_geom['point'].str.contains("bp"), 'point']
    bp_jrf_names = [[i+'_jrfx', i+'_jrfy', i+'_jrfz', i+'_jrfmag'] for i in bp_ref]
    jrf.columns = [item for sublist in bp_jrf_names for item in sublist]
    freac = pd.concat([gape, bf_y, jrf], axis=1, ignore_index=False)

    # muscle forces
    muscle_force = pd.DataFrame(fmt)
    muscle_force.columns = df_muscle['muscle']
    muscle_force = muscle_force.groupby(level=0, axis=1).sum() 
    muscle_force['total_fm'] = muscle_force.sum(axis=1)
    muscle_force = pd.concat([gape, muscle_force], axis=1, join="inner")
    
    # muscle components
    muscle_resultant = pd.DataFrame(Fm_cp_sum)
    muscle_resultant.columns = ['FMx', 'FMy', 'FMz']
    muscle_resultant = pd.concat([gape, muscle_resultant], axis=1, join="inner")
    
    # muscle strain
    muscle_strain = pd.DataFrame(lmt_strain)
    muscle_strain.columns = df_muscle['strand']
    muscle_strain = pd.concat([gape, muscle_strain], axis=1, join="inner")

    # Joint moment
    muscle_mom = M[:,:,2]
    muscle_mom = pd.DataFrame(muscle_mom)
    muscle_mom.columns = df_muscle['muscle']
    muscle_mom = muscle_mom.groupby(level=0, axis=1).sum()
    muscle_mom['total'] = muscle_mom.sum(axis=1)
    muscle_mom = pd.concat([gape, muscle_mom], axis=1, join="inner")

    # Muscle total moment arm
    muscle_mom_arm = pd.Series(muscle_mom['total']/muscle_force['total_fm'], name='total_muscle_moment_arm')
    outlever_length = pd.Series(-1*muscle_mom['total']/freac.iloc[:,3], name='outlever_length')
    muscle_mom_arm = pd.concat([gape, muscle_mom_arm, outlever_length], axis=1, join="inner")
    
    # # # muscle relative contribution to joint moment
    # muscle_mom_rel = M[:,:,2]/M_sum[:,None]
    # muscle_mom_rel = pd.DataFrame(muscle_mom_rel)
    # muscle_mom_rel.columns = df_muscle['muscle']
    # muscle_mom_rel = muscle_mom_rel.groupby(level=0, axis=1).sum() 
    # muscle_mom_rel = pd.concat([gape, muscle_mom_rel], axis=1, join="inner")

    return freac, muscle_force, muscle_resultant, muscle_strain, muscle_mom, muscle_mom_arm