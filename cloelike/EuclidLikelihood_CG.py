# General Python imports
import numpy as np
import copy

class EuclidLikelihood_CG:

    def __init__(self, data: dict, settings: dict,
                 Background: type, Perturbations: type,
                 HaloStatistics: type, SelectionFunction: type,
                 Profile: type, HaloClustering: type,
                 HaloCovariance: type, ClusterStatistics: type,
                 Pk_fid: np.ndarray,
                 ):
        r""" Class constructor
        Parameters
        ----------
        data: dict
            Data dictionary
        settings: dict
            Settings dictionary
        Background: type
            Protocol-consistent Background class type
        Perturbations: type
            Protocol-consistent Perturbations class type
        HaloStatistics: type
            Protocol-consistent HaloStatistics class type
        SelectionFunction: type
            Protocol-consistent SelectionFunction class type
        Profile: type
            Protocol-consistent Profile class type
        HaloClustering: type
            Protocol-consistent HaloClustering class type
        HaloCovariance: type
            Protocol-consistent HaloCovariance class type
        ClusterStatistics: type
            Protocol-consistent ClusterStatistics class type
        Pk_fid: type
            Pre-computed fiducial power spectrum ndarray type                                    
        """
       
        self.Background = Background
        self.Perturbations = Perturbations
        self.HaloStatistics = HaloStatistics
        self.SelectionFunction = SelectionFunction
        self.Profile = Profile
        self.HaloClustering = HaloClustering
        self.HaloCovariance = HaloCovariance
        self.ClusterStatistics = ClusterStatistics
        
        self.Pk_fid = Pk_fid

        self._prepare(data)

    def _prepare(self, data:type):
        r"""Arrange data vectors and covariance matrices in format required
        for :math:`\chi^2` calculation
        """

        self._arrange_CG_data(data)
        self._arrange_CG_cov(data)


    def _arrange_CG_data(self, data: dict):
        """Create CG Data

        Arranges the data vector for the likelihood into its final format

        Returns
        -------
        datavec: list             
        """        
        data_dict = data

        datavec_CC = np.zeros(data_dict['CG_CC'].shape[0]*data_dict['CG_CC'].shape[1])
        n = -1
        for i in range(data_dict['CG_CC'].shape[0]):
            for j in range(data_dict['CG_CC'].shape[1]):
                n = n + 1
                datavec_CC[n] = data_dict['CG_CC'][i][j]

        datavec_MoR = np.zeros(data_dict['CG_MoR'].shape[0]*data_dict['CG_MoR'].shape[1]*data_dict['CG_MoR'].shape[2])        
        n = -1
        for i in range(data_dict['CG_MoR'].shape[0]):
            for j in range(data_dict['CG_MoR'].shape[1]):
                for k in range(data_dict['CG_MoR'].shape[2]):
                    n = n + 1
                    datavec_MoR[n] = data_dict['CG_MoR'][i][j][k]        
        datavec_Cxi2 = np.zeros(data_dict['CG_xi2'].shape[0]*data_dict['CG_xi2'].shape[1]*data_dict['CG_xi2'].shape[2])
        n = -1
        for i in range(data_dict['CG_xi2'].shape[0]):
            for j in range(data_dict['CG_xi2'].shape[1]):
                for k in range(data_dict['CG_xi2'].shape[2]):
                    n = n + 1
                    datavec_Cxi2[n] = data_dict['CG_xi2'][i][j][k]
        
        self.CGCCdatafinal = datavec_CC
        self.CGMoRdatafinal = datavec_MoR
        self.CGxi2datafinal = datavec_Cxi2
        
    def _arrange_CG_cov(self, data: dict):
        """Create CG Cov

        Arranges the covariance for the likelihood into its final format

        Returns
        -------
        covfull: float N x N matrix            
        """   
        data_dict = data
        
        covfull_CC = np.zeros((data_dict['CG_cov_CC'].shape[0]*data_dict['CG_cov_CC'].shape[1]))
        n = -1
        for i in range(data_dict['CG_cov_CC'].shape[0]):
            for j in range(data_dict['CG_cov_CC'].shape[1]):
                n = n + 1
                covfull_CC[n] = data_dict['CG_cov_CC'][i][j]

        covfull_MoR = np.zeros(data_dict['CG_MoR'].shape[0]*data_dict['CG_MoR'].shape[1]*data_dict['CG_MoR'].shape[2])
        n = -1
        for i in range(data_dict['CG_MoR'].shape[0]):
            for j in range(data_dict['CG_MoR'].shape[1]):
                for k in range(data_dict['CG_MoR'].shape[2]):
                    n = n + 1
                    covfull_MoR[n] = data_dict['CG_cov_MoR'][i][j][k]

        covfull_Cxi2 = np.zeros(data_dict['CG_xi2'].shape[0]*data_dict['CG_xi2'].shape[1]*data_dict['CG_xi2'].shape[2])
        n = -1
        for i in range(data_dict['CG_xi2'].shape[0]):
            for j in range(data_dict['CG_xi2'].shape[1]):
                for k in range(data_dict['CG_xi2'].shape[2]):
                    n = n + 1
                    covfull_Cxi2[n] = data_dict['CG_cov_xi2'][i][j][k]
        
        self.CGinvcovCCfinal = covfull_CC
        self.CGinvcovMoRfinal =  covfull_MoR
        self.CGinvcovCxi2final = covfull_Cxi2     

    def get_CG_theory_vector(self, parameters: dict, settings: dict):
        """Create CG Theory

        Obtains the theory for the likelihood.

        parameters
        ----------
        dictionary: dict
            cosmology dictionary from the Cosmology class
            which is updated at each sampling step

        settings
        ----------
        dictionary: dict
            settings dictionary 

        Returns
        -------
        theoryvec: list
            returns the theory array with same indexing/format as the data
        """
        background = self.Background(
            H0=parameters['H0'], Omega_cdm0=parameters['Omega_cdm0'],
            Omega_b0=parameters['Omega_b0'], Omega_k0=parameters['Omega_k0'],
            w0=parameters['w0'], wa=parameters['wa'],  ns=parameters['ns'],
            As=parameters['As'], mnu=parameters['mnu'],
            gamma_MG=parameters['gamma_MG'])
            
        perturbations = self.Perturbations(background, np.linspace(0.0, 2.0, 40))
        perturbations_fid = self.Pk_fid

        HS = self.HaloStatistics(perturbations, zed=settings['zed'], k=settings['k'], overdensity_type=settings['overdensity_type'])

        profile = self.Profile(HS,k=settings['k'], zed=settings['zed'], r_interp=settings['r_interp'],       
        two_halo=settings['two_halo'], offcentering=settings['offcentering'], rms_off=settings['rms_off'],
        f_off=settings['f_off'], trunc_fact=settings['trunc_fact'], zs_max=settings['zs_max'],
        mean_nz=settings['mean_nz'], sigma_nz=settings['sigma_nz'], alpha_nz=settings['alpha_nz'])

        selectionFunction = self.SelectionFunction(A_l=parameters['A_l'], B_l=parameters['B_l'], C_l=parameters['C_l'],
                 sig_A_l=parameters['sig_A_l'], sig_B_l=parameters['sig_B_l'], sig_C_l=parameters['sig_C_l'], 
                 sig_lambda_norm=parameters['sig_lambda_norm'], sig_lambda_z=parameters['sig_lambda_z'], 
                 sig_lambda_exponent=parameters['sig_lambda_exponent'], sig_z_z=parameters['sig_z_z'],
                 sig_z_lambda=parameters['sig_z_lambda'])

        haloClustering = self.HaloClustering(perturbations,perturbations_fid,selectionFunction,k=settings['k'])

        covariance = self.HaloCovariance(perturbations,
                            area=settings['area'], nbins_zob=len(settings['zed_obs_edges']), 
                            k=settings['k'])

        clusterStatistics = self.ClusterStatistics(perturbations, HS, selectionFunction, profile, haloClustering, covariance,
                            zed_obs_edges=settings['zed_obs_edges'], Lambda_obs_edges=settings['Lambda_obs_edges'],
                            Rad_obs_edges=settings['Rad_obs_edges'], Lambda_obs_Cxi2_edges=settings['Lambda_obs_Cxi2_edges'],
                            Rad_obs_Cxi2_edges=settings['Rad_obs_Cxi2_edges'], zed_obs_Cxi2_edges=settings['zed_obs_Cxi2_edges'],
                            halo_concentration=settings['halo_concentration'], k=settings['k'], Mass=settings['Mass'],
                            Lambda=settings['Lambda'],zed=settings['zed'], area=settings['area'],
                            CG_like_selection=settings['CG_like_selection'], CG_xi2_cov_selection=settings['CG_xi2_cov_selection'],
                            bias=settings['bias'], neutrino_cdm=settings['neutrino_cdm'])
                  
        self.theoryvecbuf = clusterStatistics.N_zbin_Lbin_Rbin()
        
        if settings['CG_like_selection'] in ['CC','CC_CWL','CC_Cxi2','CC_CWL_Cxi2']:
           theoryvec_buf = copy.deepcopy(self.theoryvecbuf[0])
           theoryvec_CC = np.zeros( theoryvec_buf.shape[0]*theoryvec_buf.shape[1] )
           n = -1
           for i in range(theoryvec_buf.shape[0]):
              for j in range(theoryvec_buf.shape[1]):
                n = n + 1
                theoryvec_CC[n] = theoryvec_buf[i][j]

        if settings['CG_like_selection'] in ['CC_CWL','CC_CWL_Cxi2']:
           theoryvec_buf = copy.deepcopy(self.theoryvecbuf[1])
           theoryvec_MoR = np.zeros( theoryvec_buf.shape[0]*theoryvec_buf.shape[1]*theoryvec_buf.shape[2])
           n = -1
           for i in range(theoryvec_buf.shape[0]):
              for j in range(theoryvec_buf.shape[1]):
               for k in range(theoryvec_buf.shape[2]):
                n = n + 1
                theoryvec_MoR[n] = theoryvec_buf[i][j][k]

        if settings['CG_like_selection'] in ['CC_Cxi2','CC_CWL_Cxi2']:
           theoryvec_buf = copy.deepcopy(self.theoryvecbuf[2])
           theoryvec_Cxi2 = np.zeros( theoryvec_buf.shape[0]*theoryvec_buf.shape[1]*theoryvec_buf.shape[2])
           n = -1
           for i in range(theoryvec_buf.shape[0]):
              for j in range(theoryvec_buf.shape[1]):
               for k in range(theoryvec_buf.shape[2]):
                n = n + 1
                theoryvec_Cxi2[n] = theoryvec_buf[i][j][k]
        
        if settings['CG_like_selection'] == 'CC':
        	return theoryvec_CC
        elif settings['CG_like_selection'] == 'CC_CWL':
        	return theoryvec_CC, theoryvec_MoR
        elif settings['CG_like_selection'] == 'CC_Cxi2':
        	return theoryvec_CC, theoryvec_Cxi2
        elif settings['CG_like_selection'] == 'CC_CWL_Cxi2':
        	return theoryvec_CC, theoryvec_MoR, theoryvec_Cxi2

    def get_CG_theory_covariance_vector(self, settings: dict):
        """Create CG cov Theory

        Obtains the cov theory for the likelihood.

        settings
        ----------
        dictionary: dict
            settings dictionary 
            
        Returns
        -------
        theoryvec: list
            returns the covariance theory array with same indexing/format as the data
        """       
        if settings['CG_xi2_cov_selection'] in ['covCC','covCC_covCxi2']:
           theoryvec_cov_buf = copy.deepcopy(self.theoryvecbuf[3])
           theoryvec_CC_cov = np.zeros(( theoryvec_cov_buf.shape[0]*theoryvec_cov_buf.shape[2],theoryvec_cov_buf.shape[1]*theoryvec_cov_buf.shape[3] ))        
           for i in range(theoryvec_cov_buf.shape[0]):
               for j in range(theoryvec_cov_buf.shape[1]):
                   for k in range(theoryvec_cov_buf.shape[2]):
                       for l in range(theoryvec_cov_buf.shape[3]):
                           theoryvec_CC_cov[i*theoryvec_cov_buf.shape[2]+k][j*theoryvec_cov_buf.shape[3]+l] = theoryvec_cov_buf[i,j,k,l]
         
        if settings['CG_xi2_cov_selection'] in ['covCxi2','covCC_covCxi2']:
           theoryvec_cov_buf = copy.deepcopy(self.theoryvecbuf[4])
           theoryvec_Cxi2_cov = np.zeros(( theoryvec_cov_buf.shape[0]*theoryvec_cov_buf.shape[2]*theoryvec_cov_buf.shape[4],theoryvec_cov_buf.shape[1]*theoryvec_cov_buf.shape[3]*theoryvec_cov_buf.shape[5] ))        
           for i in range(theoryvec_cov_buf.shape[0]):
               for j in range(theoryvec_cov_buf.shape[1]):
                   for k in range(theoryvec_cov_buf.shape[2]):
                       for l in range(theoryvec_cov_buf.shape[3]):
                           for m in range(theoryvec_cov_buf.shape[4]):
                               for n in range(theoryvec_cov_buf.shape[5]):
                                   theoryvec_Cxi2_cov[i*theoryvec_cov_buf.shape[2]*theoryvec_cov_buf.shape[4] + \
                                                 k*theoryvec_cov_buf.shape[4] + \
                                                 m,j*theoryvec_cov_buf.shape[3]*theoryvec_cov_buf.shape[5] + \
                                                 l*theoryvec_cov_buf.shape[5]+n] = theoryvec_cov_buf[i,j,k,l,m,n]
                           
        if settings['CG_xi2_cov_selection'] == 'covCC':
           return np.linalg.inv(theoryvec_CC_cov) 
        if settings['CG_xi2_cov_selection'] == 'covCxi2':
           return np.linalg.inv(theoryvec_Cxi2_cov) 
        if settings['CG_xi2_cov_selection'] == 'covCC_covCxi2':
           return np.linalg.inv(theoryvec_CC_cov),np.linalg.inv(theoryvec_Cxi2_cov)        

    def loglike(self, parameters: dict, settings: dict):
        r""" Log-likelihood of GC probe
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        loglike: float
            Log-likelihood
        """

        if settings['CG_like_selection'] == 'CC':
            CGCCthvec = self.get_CG_theory_vector(
                             parameters, settings)               
            dmt = self.CGCCdatafinal - CGCCthvec
            if settings['CG_xi2_cov_selection'] == 'covCC':
               CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings) 
               loglike_clusters = -0.5 * np.dot(
                np.dot(dmt, CGinvcovCCfinal), dmt)
            elif settings['CG_xi2_cov_selection'] == 'CGnonanalcov':                
               loglike_clusters = -0.5 * np.sum((dmt * CGinvcovCCfinal)**2.0)               
            else :
                 raise ValueError(
                r"Choose CG covariance selection ''CGnonanalcov'' or ''covCC'' ")            
        elif settings['CG_like_selection'] == 'CC_CWL':
            createCGtheory = self.get_CG_theory_vector(
                             parameters, settings) 
            CGCCthvec = createCGtheory[0] 
            CGMoRthvec = createCGtheory[1]
            dmtCC = self.CGCCdatafinal - CGCCthvec                             
            dmtMoR = self.CGMoRdatafinal - CGMoRthvec            
            if settings['CG_xi2_cov_selection'] == 'covCC':
               CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)                      
               loglike_clusters = -0.5 * np.dot(
                np.dot(dmtCC, CGinvcovCCfinal), dmtCC) \
                       -0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0)                       
            elif settings['CG_xi2_cov_selection'] == 'CGnonanalcov': 
               loglike_clusters = -0.5 * np.sum((dmtCC * CGinvcovCCfinal)**2.0) \
                       -0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0)
            else :
                 raise ValueError(
                r"Choose CG covariance selection ''CGnonanalcov'' or ''covCC'' ")           
        elif settings['CG_like_selection'] == 'CC_Cxi2':
            createCGtheory = self.get_CG_theory_vector(
                             parameters, settings)                             
            CGCCthvec = createCGtheory[0]            
            CGxi2thvec = createCGtheory[1]                      
            dmtCC = self.CGCCdatafinal - CGCCthvec                       
            dmtCxi2 = self.CGxi2datafinal - CGxi2thvec
            if settings['CG_xi2_cov_selection'] == 'covCC':
               CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)                      
               loglike_clusters = -0.5 * np.dot(
                np.dot(dmtCC, CGinvcovCCfinal), dmtCC) \
                       -0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final)**2.0)
            elif settings['CG_xi2_cov_selection'] == 'covCxi2':
               xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)                      
               loglike_clusters = -0.5 * np.sum((dmtCC * CGinvcovCCfinal)**2.0) \
                                  -0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
            elif settings['CG_xi2_cov_selection'] == 'covCC_covCxi2':
               CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)[0]
               xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)[1]
               loglike_clusters = -0.5 * np.dot(
                np.dot(dmtCC, CGinvcovCCfinal), dmtCC) \
                                  -0.5 * np.dot(
                np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
            elif settings['CG_xi2_cov_selection'] == 'CGnonanalcov':
               loglike_clusters = -0.5 * np.sum((dmtCC * CGinvcovCCfinal)**2.0) \
                       -0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final)**2.0)
            else :
                 raise ValueError(
                r"Choose CG covariance selection ''CGnonanalcov'' or ''covCxi2'' or ''covCC'' or ''covCC_covCxi2'' ")            
        elif settings['CG_like_selection'] == 'CC_CWL_Cxi2':
            createCGtheory = self.get_CG_theory_vector(
                             parameters, settings)               
            CGCCthvec = createCGtheory[0]                             
            CGMoRthvec = createCGtheory[1]            
            CGxi2thvec = createCGtheory[2]                      
            dmtCC = self.CGCCdatafinal - CGCCthvec
            dmtMoR = self.CGMoRdatafinal - CGMoRthvec                     
            dmtCxi2 = self.CGxi2datafinal - CGxi2thvec
            if settings['CG_xi2_cov_selection'] == 'covCC':
               CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)                      
               loglike_clusters = -0.5 * np.dot(
                np.dot(dmtCC, CGinvcovCCfinal), dmtCC) \
                       -0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0) \
                       -0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final)**2.0)
            elif settings['CG_xi2_cov_selection'] == 'covCxi2':
               xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)                      
               loglike_clusters = -0.5 * np.sum((dmtCC * CGinvcovCCfinal)**2.0) \
                       -0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0) \
                       -0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
            elif settings['CG_xi2_cov_selection'] == 'covCC_covCxi2':
               CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)[0]
               xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)[1]                      
               loglike_clusters = -0.5 * np.dot(
                np.dot(dmtCC, CGinvcovCCfinal), dmtCC) \
                       -0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0) \
                       -0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)                                   
            elif settings['CG_xi2_cov_selection'] == 'CGnonanalcov': 
               loglike_clusters = -0.5 * np.sum((dmtCC * CGinvcovCCfinal)**2.0) \
                       -0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal)**2.0) \
                       -0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final)**2.0)                 
            else :
                 raise ValueError(
                r"Choose CG covariance selection ''CGnonanalcov'' or ''covCxi2'' or ''covCC'' or ''covCC_covCxi2''")
        else:
            raise ValueError(
                r"Choose CG like selection ''CC'' or ''CC_CWL'' or ''CC_Cxi2'' or ''CC_CWL_Cxi2''")

        return loglike_clusters
