# General Python imports
import numpy as np
import copy


class EuclidLikelihood_CG:

    def __init__(
        self,
        data: dict,
        settings: dict,
        _hmo_pars: dict,
        _sel_pars: dict,
        Background: type,
        Perturbations: type,
        Halo_Mass_Observable: type,        
        SelectionFunction: type,
        MatterStatistics: type,
        HaloAbundance: type,
        HaloProfile: type,
        HaloClustering: type,
        HaloCovariance: type,
        ClusterStatisticsModeling: type,
        ClusterCounts: type,
        ClusterWeakLensing: type,
        ClusterClustering: type,
        background_fid: type
    ):
        r"""Class constructor
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
        HaloAbundance: type
            Protocol-consistent HaloAbundance class type
        Halo_Mass_Observable: type
            Protocol-consistent SelectionFunction class type            
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
        self.Halo_Mass_Observable = Halo_Mass_Observable
        self.SelectionFunction = SelectionFunction
        self.MatterStatistics = MatterStatistics
        self.HaloAbundance = HaloAbundance
        self.HaloProfile = HaloProfile
        self.HaloClustering = HaloClustering
        self.HaloCovariance = HaloCovariance
        self.ClusterStatisticsModeling = ClusterStatisticsModeling
        self.ClusterCounts = ClusterCounts
        self.ClusterWeakLensing = ClusterWeakLensing
        self.ClusterClustering = ClusterClustering

        self.background_fid = background_fid
        
        self.derived = {}

        self._prepare(data)

    def _prepare(self, data: type):
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

        datavec_CC = np.zeros(data_dict["CG_CC"].shape[0] * data_dict["CG_CC"].shape[1])
        n = -1
        for i in range(data_dict["CG_CC"].shape[0]):
            for j in range(data_dict["CG_CC"].shape[1]):
                n = n + 1
                datavec_CC[n] = data_dict["CG_CC"][i][j]

        datavec_MoR = np.zeros(
            data_dict["CG_MoR"].shape[0]
            * data_dict["CG_MoR"].shape[1]
            * data_dict["CG_MoR"].shape[2]
        )
        n = -1
        for i in range(data_dict["CG_MoR"].shape[0]):
            for j in range(data_dict["CG_MoR"].shape[1]):
                for k in range(data_dict["CG_MoR"].shape[2]):
                    n = n + 1
                    datavec_MoR[n] = data_dict["CG_MoR"][i][j][k]
        datavec_Cxi2 = np.zeros(
            data_dict["CG_xi2"].shape[0]
            * data_dict["CG_xi2"].shape[1]
            * data_dict["CG_xi2"].shape[2]
        )
        n = -1
        for i in range(data_dict["CG_xi2"].shape[0]):
            for j in range(data_dict["CG_xi2"].shape[1]):
                for k in range(data_dict["CG_xi2"].shape[2]):
                    n = n + 1
                    datavec_Cxi2[n] = data_dict["CG_xi2"][i][j][k]

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

        covfull_CC = np.zeros(
            (data_dict["CG_cov_CC"].shape[0] * data_dict["CG_cov_CC"].shape[1])
        )
        n = -1
        for i in range(data_dict["CG_cov_CC"].shape[0]):
            for j in range(data_dict["CG_cov_CC"].shape[1]):
                n = n + 1
                covfull_CC[n] = data_dict["CG_cov_CC"][i][j]

        covfull_MoR = np.zeros(
            data_dict["CG_MoR"].shape[0]
            * data_dict["CG_MoR"].shape[1]
            * data_dict["CG_MoR"].shape[2]
        )
        n = -1
        for i in range(data_dict["CG_MoR"].shape[0]):
            for j in range(data_dict["CG_MoR"].shape[1]):
                for k in range(data_dict["CG_MoR"].shape[2]):
                    n = n + 1
                    covfull_MoR[n] = data_dict["CG_cov_MoR"][i][j][k]

        covfull_Cxi2 = np.zeros(
            data_dict["CG_xi2"].shape[0]
            * data_dict["CG_xi2"].shape[1]
            * data_dict["CG_xi2"].shape[2]
        )
        n = -1
        for i in range(data_dict["CG_xi2"].shape[0]):
            for j in range(data_dict["CG_xi2"].shape[1]):
                for k in range(data_dict["CG_xi2"].shape[2]):
                    n = n + 1
                    covfull_Cxi2[n] = data_dict["CG_cov_xi2"][i][j][k]

        self.CGinvcovCCfinal = 1.0/covfull_CC
        self.CGinvcovMoRfinal = 1.0/covfull_MoR
        self.CGinvcovCxi2final = 1.0/covfull_Cxi2

    def get_CG_theory_vector(self, parameters: dict, _hmo_pars: dict, _sel_pars: dict, settings: dict):
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
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=parameters["w0"],
            wa=parameters["wa"],
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
            N_mnu=parameters['N_mnu']
        )

        perturbations = self.Perturbations(background, np.linspace(0.0, 2.0, 80))
#        perturbations_fid = self.Pk_fid
        self.derived["sigma8_0"] = perturbations.sigma8_0()

        halo_mass_observable = self.Halo_Mass_Observable(
            A_l=_hmo_pars["A_l"],
            B_l=_hmo_pars["B_l"],
            C_l=_hmo_pars["C_l"],
            sig_A_l=_hmo_pars["sig_A_l"],
            sig_B_l=_hmo_pars["sig_B_l"],
            sig_C_l=_hmo_pars["sig_C_l"],
        )
        selectionfunction_counts = self.SelectionFunction(
            sig_lambda_norm=_sel_pars["sig_lambda_norm"],
            sig_lambda_z=_sel_pars["sig_lambda_z"],
            sig_lambda_exponent=_sel_pars["sig_lambda_exponent"],
            sig_z_z=_sel_pars["sig_z_z"],
            sig_z_lambda=_sel_pars["sig_z_lambda"],
            halo_mass_observable=halo_mass_observable,
            lambda_tab_integ=[31, 31, 31, 51],
            z_tab_integ=31
        )
        
        selectionfunction_lensing = self.SelectionFunction(
            sig_lambda_norm=_sel_pars["sig_lambda_norm"],
            sig_lambda_z=_sel_pars["sig_lambda_z"],
            sig_lambda_exponent=_sel_pars["sig_lambda_exponent"],
            sig_z_z=_sel_pars["sig_z_z"],
            sig_z_lambda=_sel_pars["sig_z_lambda"],
            halo_mass_observable=halo_mass_observable,
            lambda_tab_integ=[31, 31, 31, 51],
            z_tab_integ=31
        )
        
        selectionfunction_clustering = self.SelectionFunction(
            sig_lambda_norm=_sel_pars["sig_lambda_norm"],
            sig_lambda_z=_sel_pars["sig_lambda_z"],
            sig_lambda_exponent=_sel_pars["sig_lambda_exponent"],
            sig_z_z=_sel_pars["sig_z_z"],
            sig_z_lambda=_sel_pars["sig_z_lambda"],
            halo_mass_observable=halo_mass_observable,
            lambda_tab_integ=[31, 51],
            z_tab_integ=31
        )

        matterstatistics = self.MatterStatistics(
            perturbations,
            z=settings["zed"],
            k=settings["k"],
        )

        haloabundance = self.HaloAbundance(
            matter_statistics = matterstatistics 
        )

        haloprofile = self.HaloProfile(
            matterstatistics,
            overdensity_type=settings["overdensity_type"],
            overdensity=settings["overdensity"],
            z=settings["zed"],
            zs_max=settings["zs_max"],
            mean_nz=settings["mean_nz"],
            sigma_nz=settings["sigma_nz"],
            alpha_nz=settings["alpha_nz"],            
        )

        haloclustering = self.HaloClustering(
            matterstatistics,  self.background_fid
        )

        halocovariance = self.HaloCovariance(
            perturbations,
            area=settings["area"],
            nbins_zob=len(settings["zed_obs_edges"]),
            k=settings["k"],
            z_tab_integ=31
        )

        integ_ztrue_arr_new = settings["zed"].copy()
        integ_ztrue_arr_new[0] += 1.0e-10
        integ_ztrue_arr_new[-1] -= 1.0e-10

        integ_k_arr_new = settings["k"].copy()
        integ_k_arr_new[0] += 1.0e-10
        integ_k_arr_new[-1] -= 1.0e-10

        cluster_statistics_modeling = self.ClusterStatisticsModeling(
                haloabundance,
                integ_k_arr=integ_k_arr_new,
                integ_mass_arr=settings["Mass"],
                integ_lambda_true_arr=settings["Lambda"],
                integ_ztrue_arr=integ_ztrue_arr_new,
                area=settings["area"],
            )

        cluster_counts_statistics = self.ClusterCounts(
                cluster_statistics_modeling,
                halocovariance,
                selectionfunction_counts,
            )

        # To avoid SyntaxError: positional argument follows keyword argument

        halo_concentration = settings["halo_concentration"]

        cluster_wl_statistics = self.ClusterWeakLensing(
                cluster_statistics_modeling,
                haloprofile,
                halo_concentration,
                selectionfunction_lensing,
            )

        cluster_clustering_statistics = self.ClusterClustering(
                cluster_statistics_modeling,
                haloclustering,
                selectionfunction_clustering,
            )

        cluster_counts, counts_intermediate_integration_products = (
        cluster_counts_statistics.get_NC(
            z_obs_edges=settings["zed_obs_edges"],
            lambda_obs_edges=settings["Lambda_obs_edges"],
        )
    )

        cov_cluster_counts = cluster_counts_statistics.get_NC_covariance(
        settings["zed_obs_edges"],
        cluster_counts,
        counts_intermediate_integration_products["window_lambda_obs"],
        counts_intermediate_integration_products["window_z_obs"]
    )

        gt_mean_values = cluster_wl_statistics.get_gt(
        z_obs_edges=settings["zed_obs_edges"],
        lambda_obs_edges=settings["Lambda_obs_edges"],
        radius_edges=settings["Rad_obs_edges"],
    )

        cluster_clustering, clustering_intermediate_integration_products = (
        cluster_clustering_statistics.get_xi(
            lambda_obs_edges=settings["Lambda_obs_Cxi2_edges"],
            radius_edges=settings["Rad_obs_Cxi2_edges"],
            z_obs_edges=settings["zed_obs_Cxi2_edges"],
        )
    )

        cov_cluster_clustering = cluster_clustering_statistics.get_xi_covariance(
        clustering_intermediate_integration_products["pk_mean_values"],
        clustering_intermediate_integration_products["radial_shell_window"],
        clustering_intermediate_integration_products["radial_shell_volume"],
        clustering_intermediate_integration_products["window_z_obs"],
        clustering_intermediate_integration_products["cluster_counts"],
    )

        self.theoryvecbuf = []
        self.theoryvecbuf.append(cluster_counts)
        self.theoryvecbuf.append(gt_mean_values)
        self.theoryvecbuf.append(cluster_clustering)
        self.theoryvecbuf.append(cov_cluster_counts)
        self.theoryvecbuf.append(cov_cluster_clustering)

        if settings["CG_like_selection"] in ["CC", "CC_CWL", "CC_Cxi2", "CC_CWL_Cxi2"]:
            theoryvec_buf = copy.deepcopy(self.theoryvecbuf[0])
            theoryvec_CC = np.zeros(theoryvec_buf.shape[0] * theoryvec_buf.shape[1])
            n = -1
            for i in range(theoryvec_buf.shape[0]):
                for j in range(theoryvec_buf.shape[1]):
                    n = n + 1
                    theoryvec_CC[n] = theoryvec_buf[i][j]

        if settings["CG_like_selection"] in ["CC_CWL", "CC_CWL_Cxi2"]:
            theoryvec_buf = copy.deepcopy(self.theoryvecbuf[1])
            theoryvec_MoR = np.zeros(
                theoryvec_buf.shape[0] * theoryvec_buf.shape[1] * theoryvec_buf.shape[2]
            )
            n = -1
            for i in range(theoryvec_buf.shape[0]):
                for j in range(theoryvec_buf.shape[1]):
                    for k in range(theoryvec_buf.shape[2]):
                        n = n + 1
                        theoryvec_MoR[n] = theoryvec_buf[i][j][k]

        if settings["CG_like_selection"] in ["CC_Cxi2", "CC_CWL_Cxi2"]:
            theoryvec_buf = copy.deepcopy(self.theoryvecbuf[2])
            theoryvec_Cxi2 = np.zeros(
                theoryvec_buf.shape[0] * theoryvec_buf.shape[1] * theoryvec_buf.shape[2]
            )
            n = -1
            for i in range(theoryvec_buf.shape[0]):
                for j in range(theoryvec_buf.shape[1]):
                    for k in range(theoryvec_buf.shape[2]):
                        n = n + 1
                        theoryvec_Cxi2[n] = theoryvec_buf[i][j][k]

        if settings["CG_like_selection"] == "CC":
            return theoryvec_CC
        elif settings["CG_like_selection"] == "CC_CWL":
            return theoryvec_CC, theoryvec_MoR
        elif settings["CG_like_selection"] == "CC_Cxi2":
            return theoryvec_CC, theoryvec_Cxi2
        elif settings["CG_like_selection"] == "CC_CWL_Cxi2":
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
        if settings["CG_xi2_cov_selection"] in ["covCC", "covCC_covCxi2"]:
            theoryvec_cov_buf = copy.deepcopy(self.theoryvecbuf[3])
            theoryvec_CC_cov = np.zeros(
                (
                    theoryvec_cov_buf.shape[0] * theoryvec_cov_buf.shape[2],
                    theoryvec_cov_buf.shape[1] * theoryvec_cov_buf.shape[3],
                )
            )
            for i in range(theoryvec_cov_buf.shape[0]):
                for j in range(theoryvec_cov_buf.shape[1]):
                    for k in range(theoryvec_cov_buf.shape[2]):
                        for l in range(theoryvec_cov_buf.shape[3]):
                            theoryvec_CC_cov[i * theoryvec_cov_buf.shape[2] + k][
                                j * theoryvec_cov_buf.shape[3] + l
                            ] = theoryvec_cov_buf[i, j, k, l]

        if settings["CG_xi2_cov_selection"] in ["covCxi2", "covCC_covCxi2"]:
            theoryvec_cov_buf = copy.deepcopy(self.theoryvecbuf[4])
            theoryvec_Cxi2_cov = np.zeros(
                (
                    theoryvec_cov_buf.shape[0]
                    * theoryvec_cov_buf.shape[2]
                    * theoryvec_cov_buf.shape[4],
                    theoryvec_cov_buf.shape[1]
                    * theoryvec_cov_buf.shape[3]
                    * theoryvec_cov_buf.shape[5],
                )
            )
            for i in range(theoryvec_cov_buf.shape[0]):
                for j in range(theoryvec_cov_buf.shape[1]):
                    for k in range(theoryvec_cov_buf.shape[2]):
                        for l in range(theoryvec_cov_buf.shape[3]):
                            for m in range(theoryvec_cov_buf.shape[4]):
                                for n in range(theoryvec_cov_buf.shape[5]):
                                    theoryvec_Cxi2_cov[
                                        i
                                        * theoryvec_cov_buf.shape[2]
                                        * theoryvec_cov_buf.shape[4]
                                        + k * theoryvec_cov_buf.shape[4]
                                        + m,
                                        j
                                        * theoryvec_cov_buf.shape[3]
                                        * theoryvec_cov_buf.shape[5]
                                        + l * theoryvec_cov_buf.shape[5]
                                        + n,
                                    ] = theoryvec_cov_buf[i, j, k, l, m, n]

        if settings["CG_xi2_cov_selection"] == "covCC":
            return np.linalg.inv(theoryvec_CC_cov)
        if settings["CG_xi2_cov_selection"] == "covCxi2":
            return np.linalg.inv(theoryvec_Cxi2_cov)
        if settings["CG_xi2_cov_selection"] == "covCC_covCxi2":
            return np.linalg.inv(theoryvec_CC_cov), np.linalg.inv(theoryvec_Cxi2_cov)

    def loglike(self, parameters: dict, _hmo_pars: dict, _sel_pars: dict, settings: dict):
        r"""Log-likelihood of GC probe
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        loglike: float
            Log-likelihood
        """

        if settings["CG_like_selection"] == "CC":
            CGCCthvec = self.get_CG_theory_vector(parameters, _hmo_pars, _sel_pars, settings)
            dmt = self.CGCCdatafinal - CGCCthvec
            if settings["CG_xi2_cov_selection"] == "covCC":
                CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)
                loglike_clusters = -0.5 * np.dot(np.dot(dmt, CGinvcovCCfinal), dmt)
            elif settings["CG_xi2_cov_selection"] == "CGnonanalcov":
                loglike_clusters = -0.5 * np.sum((dmt * CGinvcovCCfinal) ** 2.0)
            else:
                raise ValueError(
                    r"Choose CG covariance selection ''CGnonanalcov'' or ''covCC'' "
                )
        elif settings["CG_like_selection"] == "CC_CWL":
            createCGtheory = self.get_CG_theory_vector(parameters, _hmo_pars, _sel_pars, settings)
            CGCCthvec = createCGtheory[0]
            CGMoRthvec = createCGtheory[1]
            dmtCC = self.CGCCdatafinal - CGCCthvec
            dmtMoR = self.CGMoRdatafinal - CGMoRthvec
            if settings["CG_xi2_cov_selection"] == "covCC":
                CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)
                loglike_clusters = -0.5 * np.dot(
                    np.dot(dmtCC, CGinvcovCCfinal), dmtCC
                ) - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal) ** 2.0)
            elif settings["CG_xi2_cov_selection"] == "CGnonanalcov":
                loglike_clusters = -0.5 * np.sum(
                    (dmtCC * CGinvcovCCfinal) ** 2.0
                ) - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal) ** 2.0)
            else:
                raise ValueError(
                    r"Choose CG covariance selection ''CGnonanalcov'' or ''covCC'' "
                )
        elif settings["CG_like_selection"] == "CC_Cxi2":
            createCGtheory = self.get_CG_theory_vector(parameters, _hmo_pars, _sel_pars, settings)
            CGCCthvec = createCGtheory[0]
            CGxi2thvec = createCGtheory[1]
            dmtCC = self.CGCCdatafinal - CGCCthvec
            dmtCxi2 = self.CGxi2datafinal - CGxi2thvec
            if settings["CG_xi2_cov_selection"] == "covCC":
                CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)
                loglike_clusters = -0.5 * np.dot(
                    np.dot(dmtCC, CGinvcovCCfinal), dmtCC
                ) - 0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final) ** 2.0)
            elif settings["CG_xi2_cov_selection"] == "covCxi2":
                xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)
                loglike_clusters = -0.5 * np.sum(
                    (dmtCC * CGinvcovCCfinal) ** 2.0
                ) - 0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
            elif settings["CG_xi2_cov_selection"] == "covCC_covCxi2":
                CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)[0]
                xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)[1]
                loglike_clusters = -0.5 * np.dot(
                    np.dot(dmtCC, CGinvcovCCfinal), dmtCC
                ) - 0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
            elif settings["CG_xi2_cov_selection"] == "CGnonanalcov":
                loglike_clusters = -0.5 * np.sum(
                    (dmtCC * CGinvcovCCfinal) ** 2.0
                ) - 0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final) ** 2.0)
            else:
                raise ValueError(
                    r"Choose CG covariance selection ''CGnonanalcov'' or ''covCxi2'' or ''covCC'' or ''covCC_covCxi2'' "
                )
        elif settings["CG_like_selection"] == "CC_CWL_Cxi2":
            createCGtheory = self.get_CG_theory_vector(parameters, _hmo_pars, _sel_pars, settings)
            CGCCthvec = createCGtheory[0]
            CGMoRthvec = createCGtheory[1]
            CGxi2thvec = createCGtheory[2]
            dmtCC = self.CGCCdatafinal - CGCCthvec
            dmtMoR = self.CGMoRdatafinal - CGMoRthvec
            dmtCxi2 = self.CGxi2datafinal - CGxi2thvec
            if settings["CG_xi2_cov_selection"] == "covCC":
                CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)
                loglike_clusters = (
                    -0.5 * np.dot(np.dot(dmtCC, CGinvcovCCfinal), dmtCC)
                    - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal) ** 2.0)
                    - 0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final) ** 2.0)
                )
            elif settings["CG_xi2_cov_selection"] == "covCxi2":
                xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)
                loglike_clusters = (
                    -0.5 * np.sum((dmtCC * CGinvcovCCfinal) ** 2.0)
                    - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal) ** 2.0)
                    - 0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
                )
            elif settings["CG_xi2_cov_selection"] == "covCC_covCxi2":
                CGinvcovCCfinal = self.get_CG_theory_covariance_vector(settings)[0]
                xi2invcovCCfinal = self.get_CG_theory_covariance_vector(settings)[1]
                loglike_clusters = (
                    -0.5 * np.dot(np.dot(dmtCC, CGinvcovCCfinal), dmtCC)
                    - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal) ** 2.0)
                    - 0.5 * np.dot(np.dot(dmtCxi2, xi2invcovCCfinal), dmtCxi2)
                )
            elif settings["CG_xi2_cov_selection"] == "CGnonanalcov":
                loglike_clusters = (
                    -0.5 * np.sum((dmtCC * CGinvcovCCfinal) ** 2.0)
                    - 0.5 * np.sum((dmtMoR * self.CGinvcovMoRfinal) ** 2.0)
                    - 0.5 * np.sum((dmtCxi2 * self.CGinvcovCxi2final) ** 2.0)
                )
            else:
                raise ValueError(
                    r"Choose CG covariance selection ''CGnonanalcov'' or ''covCxi2'' or ''covCC'' or ''covCC_covCxi2''"
                )
        else:
            raise ValueError(
                r"Choose CG like selection ''CC'' or ''CC_CWL'' or ''CC_Cxi2'' or ''CC_CWL_Cxi2''"
            )

        return loglike_clusters
