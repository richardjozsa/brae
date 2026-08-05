// Out-of-line definitions of brae::DeviceSimpleSolver, moved from device_simple_foam.cuh so brae_core compiles the
// SIMPLE solver bodies ONCE (every consumer TU used to re-parse ~1000 lines of inline CUDA). Bodies are verbatim;
// the class declaration (members + method signatures) stays in the header. Prerequisite for the PIMPLE solver,
// which reuses solveMomentumPredictor / correctPressureVelocity / correctTurbulence under a transient loop.
#include "device_simple_foam.cuh"

namespace brae {

    DeviceSimpleSolver::DeviceSimpleSolver(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp,
        const GeometricField<vector>& U,
        const GeometricField<scalar>& p,
        const SurfaceScalarField& phi,
        const DeviceSimpleControls& ctl,
        const GeometricField<scalar>* k,
        const GeometricField<scalar>* eps,
        const GeometricField<scalar>* nut,
        const GeometricField<scalar>* ReThetat,
        const GeometricField<scalar>* gammaInt)
        : fvp_(fvp), ctl_(ctl), nC_(m.nCells()), nIf_(m.nInternalFaces())
    {
        // cyclic (periodic) interfaces: a SEPARATE lduInterface (OF cyclicFvPatchField::updateInterfaceMatrix),
        // NOT merged into the owner-sorted LDU. Both sides of each pair are stored (symmetric coupling).
        const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
        hasCyclic_ = !cyclics.empty();
        // cyclicAMI: non-conforming interface, an area-weighted lduInterface (OF cyclicAMIFvPatchField). Built as a
        // weighted CSR stencil (device_ami); the assembly mirrors the cyclic path with the AMI interpolation.
        const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
        hasAMI_ = !amis.empty();
        if (hasCyclic_ || hasAMI_) ctl_.useGraph = false;   // V-cycle graph replay not interface-safe; minor perf only.
        dm_   = buildDeviceMesh(m, g, fvp);
        cyc_  = buildDeviceCyclic(cyclics, g, fvp);
        ami_  = buildDeviceAMI(amis);
        dbU_  = buildDeviceVectorBoundary(U, fvp, g);
        dbP_  = buildDeviceBoundary(p, fvp, g);
        wall_ = buildDeviceWallData(m, g, fvp, U);
        if (nut)
        {
            std::vector<label> cm;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                const bool calc = (fvp[pi].type != "wall") && (nut->boundary[pi]->bcCategory() == 2);
                if (calc) hasNutCalc_ = true;
                for (label i = 0; i < fvp[pi].size; ++i) cm.push_back(calc ? 1 : 0);
            }
            if (hasNutCalc_) nutCalcMask_.copyFrom(cm);
        }
        {
            // Wall-face -> boundary-face index. buildDeviceWallData walks fvp keeping type=="wall" patches;
            // DeviceBoundary walks the same fvp skipping cyclic/cyclicAMI (a wall is neither), so the wall
            // faces are a subsequence of the boundary faces and this map is just the running count.
            std::vector<label> wfb;
            label bi = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                for (label i = 0; i < fvp[pi].size; ++i, ++bi)
                    if (fvp[pi].type == "wall") wfb.push_back(bi);
            }
            if (!wfb.empty()) wfBndIdx_.copyFrom(wfb);
        }
        // all-extrapolated boundary -> gathers a cell field to boundary faces (nuEff_bnd = adjacent cell value)
        {
            std::vector<label> ty, fc;
            std::vector<scalar> ref, dc, ms;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    ty.push_back(0);
                    fc.push_back(fvp[pi].faceCells[i]);
                    ref.push_back(0.0);
                    dc.push_back(fvp[pi].deltaCoeffs[i]);
                    ms.push_back(g.magSf()[fvp[pi].start + i]);
                }
            }
            dbExtrap_.n = (int)ty.size();
            dbExtrap_.bcType.copyFrom(ty);
            dbExtrap_.refValue.copyFrom(ref);
            dbExtrap_.deltaCoeffs.copyFrom(dc);
            dbExtrap_.magSf.copyFrom(ms);
            dbExtrap_.faceCell.copyFrom(fc);
            dbExtrap_.valueFraction.copyFrom(std::vector<scalar>(ty.size(), 0.0));
            dbExtrap_.mixedMask.copyFrom(std::vector<label>(ty.size(), 0));
        }
        // any freestreamVelocity/Pressure (mixed) far-field patch present -> run the per-step valueFraction update.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->bcCategory() == 5 || p.boundary[pi]->bcCategory() == 5)
            {
                hasMixed_ = true;
                break;
            }
        // any pressureInletOutletVelocity (directionMixed) U patch present -> run its per-step updateCoeffs.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->bcCategory() == 6)
            {
                hasPiov_ = true;
                break;
            }
        // any slip/symmetry U patch -> run its per-step updateCoeffs (general non-axis-aligned normal).
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->isSymmetry())
            {
                hasSym_ = true;
                break;
            }
        // flowRateInletVelocity (massFlowRate): one masked-magSf buffer per such patch, plus the outward
        // normals over all boundary faces. Both are geometric and built once.
        {
            std::vector<scalar> nx, ny, nz;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    nx.push_back(fvp[pi].nf[i].x);
                    ny.push_back(fvp[pi].nf[i].y);
                    nz.push_back(fvp[pi].nf[i].z);
                }
            }
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (U.boundary[pi]->bcCategory() != 9) continue;
                hasFlowRate_ = true;
                std::vector<scalar> mask(nx.size(), 0.0);
                label bi = 0;
                for (std::size_t pj = 0; pj < fvp.size(); ++pj)
                {
                    if (fvp[pj].type == "cyclic" || fvp[pj].type == "cyclicAMI") continue;
                    for (label i = 0; i < fvp[pj].size; ++i, ++bi)
                        if (pj == pi) mask[bi] = fvp[pj].magSf[i];
                }
                frMagSf_.emplace_back();
                frMagSf_.back().copyFrom(mask);
                // OF re-reads flowRate_->value(t) each call; steady + constant -> the seeded value. It is
                // recovered from the seeded BC rather than re-parsed: avgU*sum(rho_seed*magSf) = -mdot.
                frPatches_.push_back(FlowRatePatch{U.boundary[pi]->flowRateValue()});
            }
            if (hasFlowRate_) { frNx_.copyFrom(nx); frNy_.copyFrom(ny); frNz_.copyFrom(nz); }
        }
        // any totalPressure p patch -> recompute its fixedValue refValue each step from the patch velocity + flux.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (p.boundary[pi]->bcCategory() == 7)
            {
                hasTotalP_ = true;
                break;
            }
        // per-boundary-face wall mask + nearWallDist y (for the true boundary nut = nutkWallFunction at walls).
        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
            std::vector<label> isW;
            std::vector<scalar> yv;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                const bool wall = (fvp[pi].type == "wall");
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    isW.push_back(wall ? 1 : 0);
                    yv.push_back(wall ? yW[pi][i] : 0.0);
                }
            }
            bndIsWall_.copyFrom(isW);
            bndY_.copyFrom(yv);
        }
        // flat boundary face centres (patch order, cyclic/cyclicAMI excluded) -- the absolute-position Cf indexing used
        // by the NVRTC coded-BC kernels (aligned to the DeviceBoundary refValue/faceCell order). Built once at setup.
        {
            std::vector<scalar> bx, by, bz;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    const vector& cf = g.Cf()[fvp[pi].start + i];
                    bx.push_back(cf.x); by.push_back(cf.y); bz.push_back(cf.z);
                }
            }
            bndCfX_.copyFrom(bx); bndCfY_.copyFrom(by); bndCfZ_.copyFrom(bz);
        }
        // adjustPhi adjustable mask (patch order = phiBnd_ order): a face is adjustable iff its U patch does NOT
        // fix the flux (zeroGradient/inletOutlet/calculated) -> OF "!fixesValue || isA<inletOutlet>".
        {
            std::vector<label> adj;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                const bool fixed = U.boundary[pi]->fixesValue();
                for (label i = 0; i < fvp[pi].size; ++i)
                    adj.push_back(fixed ? 0 : 1);
            }
            adjustMask_.copyFrom(adj);
        }
        nuBndConst_.copyFrom(std::vector<scalar>(dbExtrap_.n, ctl.nu));
        // AMG hierarchy for the pressure Laplacian (static: faceWeights = |Sf|), built from the mesh internal faces.
        // The cyclic interface is NOT in the AMG: a Galerkin coarse operator built from the internal-face restriction
        // cannot represent the long-range periodic edges (the V-cycle diverges, OF handles this with a dedicated
        // cyclicGAMGInterface agglomerated at every level). Instead, when cyclic is present the pressure is solved with
        // Jacobi-PCG over the interface-coupled fine operator (deviceLduViewCyclic), the device analog of the validated
        // host cyclicPCG. cyclicGAMGInterface is the deferred perf path; for now AMG is only used on non-cyclic cases.
        const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf_), neiInt(m.neighbour());
        std::vector<scalar> fwAMG(g.magSf().begin(), g.magSf().begin() + nIf_);   // default: geometric face area (bit-identical)
        if (std::getenv("BRAE_AMG_SOC"))   // SoC ON: weight = LAPLACIAN coupling magSf*deltaCoeffs (the real matrix strength,
        {
            const auto& dcg = g.deltaCoeffs();                                    // up to ~constant rAU) so agglomerate()'s
            for (label f = 0; f < nIf_; ++f)
                fwAMG[f] *= dcg[f];                  // strong-face filter sees the true anisotropy
        }
        amg_ = buildOrLoadAMG(ownerInt, neiInt, fwAMG, nC_, ctl_.caseDir + "/constant/polyMesh", ctl_.writeCache);

        // initial device state.
        {
            std::vector<scalar> ux(nC_), uy(nC_), uz(nC_);
            for (label c = 0; c < nC_; ++c)
            {
                ux[c]=U.internal[c].x;
                uy[c]=U.internal[c].y;
                uz[c]=U.internal[c].z;
            }
            Uk_[0].copyFrom(ux);
            Uk_[1].copyFrom(uy);
            Uk_[2].copyFrom(uz);
        }
        dp_.copyFrom(p.internal);
        phiInt_.copyFrom(phi.internal);             // internal-face flux (cyclic flux is held separately in cyc_.phi)
        // initialise the persistent cyclic-face flux. The host phi was built with the cyclic patch as a zeroGradient
        // PLACEHOLDER (makePatchField), so its cyclic boundary flux is the un-coupled, un-rotated own-cell value (and
        // for rotational it is NOT conservative). Recompute it on the device from U_init with the proper coupled +
        // rotated interpolation (deviceCyclicFlux[Rot] is conservative) so the first momentum convection + continuity
        // see the correct periodic flux. (For a rest start U=0 this is 0 either way; it matters for a swirling init.)
        if (hasCyclic_)
        {
            if (cyc_.rotational) deviceCyclicFluxRot(cyc_, Uk_[0], Uk_[1], Uk_[2]);
            else                 interfaceFlux(cyc_, Uk_[0], Uk_[1], Uk_[2]);
        }
        if (hasAMI_) interfaceFlux(ami_, Uk_[0], Uk_[1], Uk_[2]);   // conservative initial AMI flux from U_init
        {
            std::vector<scalar> o;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "cyclic" || fvp[pi].type == "cyclicAMI") continue;
                o.insert(o.end(), phi.boundary[pi].begin(), phi.boundary[pi].end());
            }
            phiBnd_.copyFrom(o);
        }
        std::vector<scalar> nv(nC_, 0.0);
        if (ctl_.turbulent)
        {
            nv = nut->internal;   // nut is the ONLY turbulence field for pure LES (Smagorinsky); read it for every model.
            if (!ctl_.les)        // RAS/DES transport scalars: pure-LES Smagorinsky is ALGEBRAIC (no k/epsilon/omega/nuTilda, no wall distance).
            {
                dbK_   = buildDeviceBoundary(*k, fvp, g);        // SA: the "k" slot holds nuTilda
                dk_.copyFrom(k->internal);
                if (!ctl_.sa)
                {
                    dbEps_ = buildDeviceBoundary(*eps, fvp, g);
                    de_.copyFrom(eps->internal);   // 2nd scalar (omega/eps); SA has none
                }
                if (ctl_.sst || ctl_.sa)
                {
                    std::vector<vector> wallOrigin;                              // nearest wall-face centre (IDDES wall-normal source)
                    y_.copyFrom(cellWallDist(m, g, fvp, ctl_.iddes ? &wallOrigin : nullptr));   // wall distance (SST F1/F2/F3; SA dTilda)
                    if (ctl_.iddes)   // IDDES filter widths (maxDeltaxyz + wall-normal spacing), uploaded once at setup
                    {
                        const std::vector<scalar> hmax = cellMaxDeltaXYZ(m);
                        hmax_.copyFrom(hmax);
                        hwn_.copyFrom(cellWallNormalSpacing(m, g, wallOrigin, hmax));
                    }
                }
                if (ctl_.lm && ReThetat && gammaInt)   // kOmegaSSTLM transition fields
                {
                    dbReThetat_ = buildDeviceBoundary(*ReThetat, fvp, g);
                    ReThetat_.copyFrom(ReThetat->internal);
                    dbGammaInt_ = buildDeviceBoundary(*gammaInt, fvp, g);
                    gammaInt_.copyFrom(gammaInt->internal);
                    gammaIntEff_.copyFrom(std::vector<scalar>(nC_, 0.0));   // OF kOmegaSSTLM ctor: gammaIntEff_ = Zero (no Pk on iter 1, before correctReThetatGammaInt updates it)
                }
            }
        }
        dnut_.copyFrom(nv);
        nuConst_.copyFrom(std::vector<scalar>(nC_, ctl_.nu));
        zeroSrc_.copyFrom(std::vector<scalar>(nC_, 0.0));
        zeroBndU_.copyFrom(std::vector<scalar>(dbU_.n, 0.0));
        ones_.copyFrom(std::vector<scalar>(nC_, 1.0));            // unit vector for the OF normFactor
        validateTurbulence();                                    // OF turbulenceModel ctor bound() + simpleFoam.C:92 validate()
    }

    void DeviceSimpleSolver::addCodedBC(const std::string& name, const std::string& code, int offset, int count, int target, bool mixed)
    {
        SolverCodedBC cbc;
        const bool vec = (target == 0);   // U is vector; p/k/second are scalar
        cbc.kernel = mixed ? (vec ? compileCodedMixedVectorBc(name, code) : compileCodedMixedScalarBc(name, code))
                           : (vec ? compileCodedVectorBc(name, code)      : compileCodedScalarBc(name, code));
        cbc.offset = offset;
        cbc.count  = count;
        cbc.target = target;
        cbc.mixed  = mixed;
        codedBCs_.push_back(std::move(cbc));
    }

    void DeviceSimpleSolver::validateTurbulence()
    {
        if (!ctl_.turbulent) return;
        const DeviceMesh& dm = dm_;
        if (ctl_.les)   // pure LES Smagorinsky: nut is algebraic (no bound(k)/bound(omega)); seed it from the initial U.
        {
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            deviceSmagorinskyNut(nC_, gradU, dm.V, ctl_.smagCoeffs, dnut_);   // nut = Ck*delta*sqrt(k_sgs)
            return;
        }
        deviceBoundField(dm, dk_, 1e-15);                              // bound(k_, kMin_)  [SA: bound(nuTilda_, 0)]
        if (!ctl_.sa) deviceBoundField(dm, de_, 1e-15);               // bound(omega_|epsilon_, ...Min_)
        // validate() -> correctNut(): recompute internal nut from the bounded fields + the initial U.
        if (ctl_.sa)
        {
            deviceNutSA(dk_, ctl_.nu, ctl_.saCoeffs.Cv1, dnut_);     // nut = nuTilda*fv1(nuTilda)
        }
        else if (ctl_.sst)
        {
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU,
                        hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.gradULimitK > 0.0) deviceCellLimitGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, ctl_.gradULimitK);   // OF grad(U) cellLimited (validate correctNut)
            DeviceBuffer<scalar> S2;
            deviceS2(gradU, nC_, S2);
            DeviceBuffer<scalar> F2;
            deviceF2(dk_, de_, y_, ctl_.nu, ctl_.ksstCoeffs, F2);
            deviceNutSST(dk_, de_, F2, S2, ctl_.ksstCoeffs, dnut_);   // nut = a1*k/max(a1*omega, b1*F2*sqrt(S2))
        }
        else
        {
            deviceNut(dk_, de_, dnut_, ctl_.keCoeffs);               // nut = Cmu*k^2/eps
        }
    }

    void DeviceSimpleSolver::correctTurbulence()
    {
        const DeviceMesh& dm = dm_;
        clearTurbulenceReport();   // OF-style report: collect this step's turbulence solves (Solving for k/omega/...)
        if (ctl_.les)   // pure LES Smagorinsky: algebraic sub-grid nut = Ck*delta*sqrt(k_sgs) from the current U. No
        {              // transport solve (report stays empty -> no "Solving for k/omega" lines), so no ddt(k/omega) either.
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            deviceSmagorinskyNut(nC_, gradU, dm.V, ctl_.smagCoeffs, dnut_);
            return;
        }
        if (ctl_.turbulent)
        {
            // transient URANS fvm::ddt(k/eps/omega/nuTilda): a per-scalar bundle (steady -> active==false -> no-op, so
            // the steady SIMPLE turbulence stays byte-for-byte). dk_=k|nuTilda (kOld_), de_=epsilon|omega (e2Old_).
            const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
            const ScalarDdt kDdt{ ddtc, &kOld_,  kOld2_.size()  ? &kOld2_  : nullptr, ddtc.cn ? &kddt0_ : nullptr };
            const ScalarDdt sDdt{ ddtc, &e2Old_, e2Old2_.size() ? &e2Old2_ : nullptr, ddtc.cn ? &e2ddt0_ : nullptr };
            const ScalarDdt reDdt{ ddtc, &ReThetatOld_, ReThetatOld2_.size() ? &ReThetatOld2_ : nullptr, ddtc.cn ? &ReThetatddt0_ : nullptr };   // kOmegaSSTLM
            const ScalarDdt giDdt{ ddtc, &gammaIntOld_, gammaIntOld2_.size() ? &gammaIntOld2_ : nullptr, ddtc.cn ? &gammaIntddt0_ : nullptr };
            if (ctl_.sa)    // one-equation: dk_ slot holds nuTilda; relaxK/limitedK/twoBykK carry the nuTilda settings
                deviceSpalartAllmarasCorrect(dm, dbU_, dbK_, Uk_[0], Uk_[1], Uk_[2], dk_, dnut_, y_, phiInt_, phiBnd_,
                                             ctl_.nu, ctl_.relaxK, ctl_.tolKE, ctl_.bounded, ctl_.limitedK, ctl_.twoBykK,
                                             ctl_.saCoeffs, ctl_.relTolKE, ctl_.bicgCheckEvery, ctl_.luK, ctl_.nonOrth,
                                             ctl_.gsK, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr, kDdt,   // nuTilda ddt (kOld_)
                                             ctl_.des, ctl_.iddes, ctl_.iddes ? &hmax_ : nullptr, ctl_.iddes ? &hwn_ : nullptr);   // SA-DDES/IDDES length-scale limiter (no-op for plain SA-RANS)
            else if (ctl_.sst)   // de_ slot holds omega; relaxEps/limitedEps/twoBykEps carry the omega-equation settings
            {
                deviceKOmegaSSTCorrect(dm, wall_, dbEps_, dbK_, dbU_, Uk_[0], Uk_[1], Uk_[2], dk_, de_, dnut_, y_,
                                       phiInt_, phiBnd_, ctl_.nu, ctl_.relaxEps, ctl_.relaxK, ctl_.tolKE, ctl_.bounded,
                                       ctl_.limitedK, ctl_.limitedEps, ctl_.twoBykK, ctl_.twoBykEps, ctl_.ksstCoeffs, ctl_.relTolKE, ctl_.bicgCheckEvery,
                                       ctl_.luK, ctl_.luEps, ctl_.nonOrth, ctl_.gradULimitK, ctl_.gsK, ctl_.gsEps, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr,
                                       ctl_.lm ? gammaIntEff_.data() : nullptr,   // LM: scale k Pk/epsilonByk by the lagged gammaIntEff
                                       static_cast<int>(ctl_.nutWall),   // near-wall G0 uses the same BC-chosen wall nut as the momentum shear
                                       ctl_.atmZ0, ctl_.atmBoundNut,   // atmNutkWallFunction roughness for the near-wall G0
                                       kDdt, sDdt, ctl_.des, ctl_.iddes, ctl_.iddes ? &hmax_ : nullptr, ctl_.iddes ? &hwn_ : nullptr,   // ddt(k)/ddt(omega) + kOmegaSSTDDES/IDDES DES limiter (no-op for RANS)
                                       compressible_ ? &th_.rho : nullptr,   // rho-weight every k/omega term, as OF's alpha*rho* does
                                       compressible_ ? &th_.mu : nullptr,   // laminar dynamic mu; F1/F2 get nu = mu/rho from it
                                       (compressible_ && wfNu_.size()) ? &wfNu_ : nullptr,   // nu at WALL faces for omegaWallFunction/G0
                                       compressible_ ? &rhoBnd_ : nullptr,   // rho_b: divU is div of the VOLUMETRIC flux, not the mass flux
                                       // Order matters: with an EXTRAPOLATED nut_b this made omega worse
                                       // (3.1e-5 -> 3.2e-3). Enabled only now that nut_b is evaluated.
                                       nutBndAll_.size() ? &nutBndAll_ : nullptr,
                                       compressible_ ? &muBnd_ : nullptr);
                if (ctl_.lm)   // Langtry-Menter: transport ReThetat + gammaInt, update gammaIntEff for next iter
                    deviceKOmegaSSTLMCorrect(dm, dbU_, dbReThetat_, dbGammaInt_, Uk_[0], Uk_[1], Uk_[2], dk_, de_, dnut_, y_,
                                             ReThetat_, gammaInt_, gammaIntEff_, phiInt_, phiBnd_, ctl_.nu, ctl_.relaxEps,
                                             ctl_.tolKE, ctl_.relTolKE, ctl_.bicgCheckEvery, ctl_.bounded, ctl_.nonOrth,
                                             ctl_.gsEps, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr, reDdt, giDdt);   // LM transition ddt
            }
            else
                deviceKEpsilonCorrect(dm, wall_, dbEps_, dbK_, dbU_, Uk_[0], Uk_[1], Uk_[2], dk_, de_, dnut_,
                                      phiInt_, phiBnd_, ctl_.nu, ctl_.relaxEps, ctl_.relaxK, ctl_.tolKE, ctl_.bounded,
                                      ctl_.limitedK, ctl_.limitedEps, ctl_.twoBykK, ctl_.twoBykEps, ctl_.keCoeffs, ctl_.relTolKE, ctl_.bicgCheckEvery,
                                      ctl_.luK, ctl_.luEps, ctl_.nonOrth, ctl_.gsK, ctl_.gsEps, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr,
                                      static_cast<int>(ctl_.nutWall),   // near-wall G0 uses the same BC-chosen wall nut as the momentum shear
                                      ctl_.atmZ0, ctl_.atmBoundNut,   // atmNutkWallFunction roughness for the near-wall G0
                                      kDdt, sDdt,   // transient fvm::ddt(k)/ddt(epsilon)  (sDdt = the 2nd-scalar bundle)
                                      compressible_ ? &th_.rho : nullptr,   // alpha*rho on every k/eps RHS term + diffusivity
                                      compressible_ ? &th_.mu : nullptr,    // laminar dynamic mu
                                      compressible_ ? &rhoBnd_ : nullptr,   // rho_b: divU from the VOLUMETRIC flux
                                      (compressible_ && wfNu_.size()) ? &wfNu_ : nullptr,   // nu at wall faces (OF nu(patchi))
                                      nutBndAll_.size() ? &nutBndAll_ : nullptr,   // nut_b -> DkEff/DepsEff(patchi)
                                      compressible_ ? &muBnd_ : nullptr);
        }
    }

    void DeviceSimpleSolver::solveMomentumPredictor(DeviceSimpleResidual& res)
    {
        const DeviceMesh& dm = dm_;
        const scalar tol = ctl_.tolU;
        // Transient fvm::ddt(U): the ONE term steady SIMPLE lacks. steadyState (SIMPLE) -> ddtc.active==false -> the two
        // deviceFvmDdt* calls below are exact no-ops (this method stays byte-for-byte SIMPLE). pimpleStep() sets the
        // scheme + rotates the old-time fields, so the transient (PIMPLE) path folds ddt into the SAME predictor.
        const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
        // nonOrthLimitP (the pressure non-orth limiter) is declared in step() -- it is read only by the pressure phase.
        const scalar nonOrthRelaxU = std::getenv("BRAE_NONORTH_RELAX_U") ? std::atof(std::getenv("BRAE_NONORTH_RELAX_U")) : 1.0;

        // inletOutlet BCs: resolve each IO face to fixedValue|zeroGradient from the PREVIOUS step's boundary flux
        // (OF updateCoeffs uses the prior corrector's phi). No-op when a boundary has no inletOutlet faces.
        deviceUpdateInletOutlet(dbU_, phiBnd_);
        deviceUpdateInletOutlet(dbP_, phiBnd_);
        if (ctl_.turbulent && !ctl_.les)   // pure LES has no k/epsilon/omega boundaries (algebraic nut)
        {
            deviceUpdateInletOutlet(dbK_, phiBnd_);
            deviceUpdateInletOutlet(dbEps_, phiBnd_);
            if (ctl_.lm)
            {
                deviceUpdateInletOutlet(dbReThetat_, phiBnd_);
                deviceUpdateInletOutlet(dbGammaInt_, phiBnd_);
            }
        }
        // mixed freestreamVelocity/Pressure: recompute the per-face valueFraction from the (lagged) flow angle.
        if (hasMixed_) deviceUpdateMixedFreestream(dbU_, dbP_, phiBnd_, Uk_[0], Uk_[1], Uk_[2],
                                                  compressible_ ? &rhoBnd_ : nullptr);   // phiBnd_ is a MASS flux
        if (hasPiov_)  deviceUpdatePressureInletOutletVelocity(dbU_, phiBnd_, Uk_[0], Uk_[1], Uk_[2]);
        if (hasSym_)   deviceUpdateSymmetry(dbU_, Uk_[0], Uk_[1], Uk_[2]);
        // totalPressure p: recompute refValue = p0 - 0.5*neg(phi)|U_b|^2 from the boundary velocity (deviceBCValue
        // reflects the just-updated U BCs above, e.g. pressureInletOutletVelocity) + the boundary flux.
        if (hasTotalP_)
        {
            DeviceBuffer<scalar> ubx, uby, ubz;
            deviceBCValue(dbU_.comp[0], Uk_[0], ubx);
            deviceBCValue(dbU_.comp[1], Uk_[1], uby);
            deviceBCValue(dbU_.comp[2], Uk_[2], ubz);
            deviceUpdateTotalPressure(dbP_, phiBnd_, ubx, uby, ubz,
                                      compressible_ ? &rhoBnd_ : nullptr);   // p in Pa -> rho-weighted dynamic head
        }
        // NVRTC device-coded BCs (codedFixedValue): run each compiled snippet over its patch faces, overwriting the
        // target field's refValue on the device from position/time/adjacent-cell value. Same updateCoeffs point as the
        // BCs above; the p/k/second boundaries set here persist into the pressure + turbulence phases of this corrector.
        for (const SolverCodedBC& cbc : codedBCs_)
        {
            if (cbc.target == 0)         // U (vector)
            {
                if (cbc.mixed)
                    launchCodedMixedVectorBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_,
                                             Uk_[0], Uk_[1], Uk_[2], dbU_.comp[0].faceCell,
                                             dbU_.comp[0].refValue, dbU_.comp[1].refValue, dbU_.comp[2].refValue,
                                             dbU_.comp[0].valueFraction, dbU_.comp[1].valueFraction, dbU_.comp[2].valueFraction);
                else
                    launchCodedVectorBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_,
                                        Uk_[0], Uk_[1], Uk_[2], dbU_.comp[0].faceCell,
                                        dbU_.comp[0].refValue, dbU_.comp[1].refValue, dbU_.comp[2].refValue);
            }
            else                         // scalar: p (1), k/nuTilda (2), second omega|epsilon (3)
            {
                DeviceBoundary& db = (cbc.target == 1) ? dbP_ : (cbc.target == 2) ? dbK_ : dbEps_;
                const DeviceBuffer<scalar>& fld = (cbc.target == 1) ? dp_ : (cbc.target == 2) ? dk_ : de_;
                if (cbc.mixed)
                    launchCodedMixedScalarBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_, fld, db.faceCell, db.refValue, db.valueFraction);
                else
                    launchCodedScalarBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_, fld, db.faceCell, db.refValue);
            }
        }

        DeviceBuffer<scalar> nuEff;
        if (compressible_)
            // muEff = mu + rho*nut. OF assembles the stress as rho*nuEff() (linearViscousStress.C), and for a
            // compressible model nuEff() = nut() + mu/rho, so rho*nuEff() = rho*nut + mu = mut + mu
            // (CompressibleTurbulenceModel::muEff). dnut_ stays KINEMATIC -- the SST solves for it, every wall
            // function compares it against nu, and alphat = rho*nut/Prt reads it -- so the rho belongs here.
            deviceHadamard(nuEff, th_.rho, dnut_);
        else
            deviceCopy(nuEff, dnut_);
        deviceAxpy(1.0, nuConst_, nuEff);   // + nu (incompressible) / + mu (compressible, nuConst_ = th_.mu)
        DeviceBuffer<scalar> nuEff_f;
        deviceInterpolate(dm, nuEff, nuEff_f);
        // nuEff at boundary FACES: turbulent uses the TRUE wall nut (nutkWallFunction), not the cell value, so the
        // wall shear matches OpenFOAM; laminar/non-wall faces use the adjacent cell value.
        // Boundary turbulent viscosity into muEff_b. OF: mut(patchi) = rho.boundaryField()[patchi]*nut(patchi),
        // so the wall nut -- which every wall function returns KINEMATIC -- is rho_b-weighted here, exactly as
        // the cell value is above. Incompressible keeps the plain sum.
        auto addWallNutToMuEff =
            [&](const DeviceBuffer<scalar>& nutB,
                DeviceBuffer<scalar>& muEffB)
            {
                deviceCopy(nutBndAll_, nutB);   // same nut: alphat_b (compressible) + DkEff/DepsEff(patchi)
                if (!compressible_)
                {
                    deviceAxpy(1.0, nutB, muEffB);
                    return;
                }
                DeviceBuffer<scalar> mutB;
                deviceHadamard(mutB, rhoBnd_, nutB);
                deviceAxpy(1.0, mutB, muEffB);
            };

        DeviceBuffer<scalar> nuEffBnd;
        if (ctl_.turbulent && ctl_.les)
        {
            // pure LES Smagorinsky: NO k-based wall function (there is no k field). Honour a velocity-based nut wall
            // function (nutUSpaldingWallFunction) if the 0/nut BC selects it; otherwise extrapolate the cell nuEff
            // (nu+nut) to the boundary (calculated/zeroGradient/fixedValue nut -> adjacent-cell sub-grid nut), exactly
            // as the laminar path does.
            if (ctl_.nutWall == NutWall::Spalding)
            {
                deviceCopy(nuEffBnd, nuBndConst_);
                deviceBoundaryNutSpalding(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu, ctl_.saCoeffs, dnutBndWall_,
                                          compressible_ ? &nuWallBnd_ : nullptr);
                deviceAxpy(1.0, dnutBndWall_, nuEffBnd);
            }
            else
                deviceBCValue(dbExtrap_, nuEff, nuEffBnd);
        }
        else if (ctl_.turbulent)
        {
            deviceCopy(nuEffBnd, nuBndConst_);
            // Wall nut chosen by the 0/nut BC TYPE (ctl_.nutWall), matching OpenFOAM, NOT the model.
            if (ctl_.sa || ctl_.nutWall == NutWall::Spalding)   // nutUSpaldingWallFunction (velocity-based Newton uTau): SA always, or the BC on any model
            {
                deviceBoundaryNutSpalding(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu, ctl_.saCoeffs, dnutBndWall_);
                addWallNutToMuEff(dnutBndWall_, nuEffBnd);
            }
            else if (ctl_.nutWall == NutWall::Blended)   // nutUBlendedWallFunction (velocity-based binomial n=4 blend) on kEps/kOmegaSST
            {
                deviceBoundaryNutBlended(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu, ctl_.keCoeffs.kappa, ctl_.keCoeffs.E, dnutBndWall_,
                                         compressible_ ? &nuWallBnd_ : nullptr);
                addWallNutToMuEff(dnutBndWall_, nuEffBnd);
            }
            else   // k-based wall nut: nutkWallFunction (smooth), or atmNutkWallFunction (rough) when ctl_.atmZ0>0
            {
                DeviceBuffer<scalar> nutBnd;
                // kEpsilon only: the 'calculated' nut patches carry Cmu*k_b^2/eps_b. The SST's nut has a
                // different expression (a1*k/max(a1*omega, b1*F2*sqrt(S2))) that needs boundary F2 and S2,
                // so it keeps the extrapolated value until that is measured and built.
                DeviceBuffer<scalar> kB, eB;
                const bool keNut = hasNutCalc_ && !ctl_.sst && !ctl_.sa && !ctl_.keCoeffs.realizable;
                if (keNut) { deviceBCValue(dbK_, dk_, kB); deviceBCValue(dbEps_, de_, eB); }
                deviceBoundaryNut(dbU_.comp[0], bndIsWall_, bndY_, dk_, dnut_, ctl_.nu, nutBnd, ctl_.keCoeffs, ctl_.atmZ0, ctl_.atmBoundNut,
                                  compressible_ ? &nuWallBnd_ : nullptr,
                                  keNut ? &nutCalcMask_ : nullptr,
                                  keNut ? &kB : nullptr,
                                  keNut ? &eB : nullptr);
                // SST: the 'calculated' patches carry a1*k_b/max(a1*om_b, b1*F2_b*sqrt(S2_b)) -- a different
                // expression from kEpsilon's, needing boundary F2 and S2. Overwrites those faces only.
                if (hasNutCalc_ && ctl_.sst)
                {
                    DeviceBuffer<scalar> kBs, omBs, gradUs;
                    deviceBCValue(dbK_, dk_, kBs);
                    deviceBCValue(dbEps_, de_, omBs);
                    deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradUs,
                                hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
                    deviceSSTNutBoundary(dbU_, kBs, omBs, y_,   // CELL wall distance (bndY_ is 0 off-wall)
                                         compressible_ ? &nuWallBnd_ : nullptr, ctl_.nu,
                                         gradUs, nC_, Uk_[0], Uk_[1], Uk_[2],
                                         nutCalcMask_, ctl_.ksstCoeffs, dnut_, nutBnd);
                }
                addWallNutToMuEff(nutBnd, nuEffBnd);
            }
        }
        else if (compressible_)
            deviceCopy(nuEffBnd, muBnd_);   // laminar compressible: mu_b = Sutherland(T_b), OF transport_.mu(patchi)
        else
            deviceBCValue(dbExtrap_, nuEff, nuEffBnd);   // laminar: adjacent-cell value
        // explicit divDevReff stress (uses the incoming U; coupled across the 3 components)
        DeviceBuffer<scalar> ddrX, ddrY, ddrZ;
        deviceDivDevReff(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], nuEff, nuEffBnd, ddrX, ddrY, ddrZ, hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr);
        DeviceBuffer<scalar>* ddr[3] = { &ddrX, &ddrY, &ddrZ };

        DeviceBuffer<scalar> pbv;
        deviceBCValue(dbP_, dp_, pbv);
        // gx/gy/gz are members (shared with the pressure phase).
        deviceGaussGrad(dm, dp_, pbv, gx, gy, gz);
        if (hasCyclic_) interfaceAddGrad(cyc_, dp_, dm.V, gx, gy, gz);   // cyclic-face contribution to grad(p) in the momentum source
        if (hasAMI_)    interfaceAddGrad(ami_, dp_, dm.V, gx, gy, gz);       // AMI-face contribution to grad(p)
        DeviceBuffer<scalar> mDiag, lD, lU, lL;   // mUp/mLo are now members (shared with the pressure phase)
        deviceDivUpwindCoeffs(dm, phiInt_, mDiag, mUp, mLo);
        deviceLaplacianCoeffs(dm, nuEff_f, lD, lU, lL, ctl_.nonOrth);
        deviceAxpy(-1.0,lD,mDiag);
        deviceAxpy(-1.0,lU,mUp);
        deviceAxpy(-1.0,lL,mLo);
        // bounded Gauss upwind: - fvm::Sp(fvc::div(phi), U). mDiag -= V*div(phi). Stabilises the rest-start
        // transient (vanishes at convergence); matches OF's bounded scheme + the CPU simpleStep.
        // div(phi) MUST include the cyclic-face flux (OF's fvc::div sums ALL faces). Without it the cyclic cells
        // see the un-cancelled cyclic flux -/+phi/V as a spurious divergence, so the bounded term injects -/+phi into the
        // diagonal, asymmetric by 2phi between the inflow/outflow cyclic sides -> asymmetric rAU/HbyA -> non-axisym p.
        if (ctl_.bounded)
        {
            DeviceBuffer<scalar> dphiM;
            deviceDiv(dm, phiInt_, phiBnd_, dphiM);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, dphiM);
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, dphiM);
            DeviceBuffer<scalar> bM;
            deviceHadamard(bM, dphiM, dm.V);
            deviceAxpy(-1.0, bM, mDiag);
        }
        // cyclic (periodic) momentum coupling M = div(phi,U) - laplacian(nuEff,U): folds the interface diagonal into
        // mDiag + builds cyc_.ifCoeff (off-diag) from the current cyclic flux; cycSumOff feeds the relax dominance.
        DeviceBuffer<scalar> cycSumOff;
        if (hasCyclic_)
        {
            interfaceAssembleMomentum(cyc_, nuEff, mDiag);
            cycSumOff.copyFrom(std::vector<scalar>(nC_, 0.0));
            interfaceOffDiagSum(cyc_, cycSumOff);
            if (cyc_.rotational) interfaceScaleImplicit(cyc_);   // per-component ifCoeffC[kk] = ifCoeff*forwardT[kk][kk]
        }
        DeviceBuffer<scalar> amiSumOff;                              // cyclicAMI momentum coupling (translational)
        if (hasAMI_)
        {
            interfaceAssembleMomentum(ami_, nuEff, mDiag);
            amiSumOff.copyFrom(std::vector<scalar>(nC_, 0.0));
            interfaceOffDiagSum(ami_, amiSumOff);
            if (ami_.rotational) interfaceScaleImplicit(ami_);   // per-component ifCoeffC[kk] = ifCoeff*forwardT[kk][kk]
        }
        if (fvoMomSp_.size()) deviceAxpy(-1.0, fvoMomSp_, mDiag);   // fvOptions implicit momentum Sp: diag -= Sp*V (== fvm::Sp)
        // explicitPorositySource (DarcyForchheimer): isotropic resistance into the diagonal (mDiag += V*tr(Cd)); the
        // anisotropic remainder is added to relaxSrc per component below. Uses the LAMINAR nu (OF mu/rho, not nuEff).
        if (por_.active) deviceFvoPorosityDiag(por_, ctl_.nu, dm.V, Uk_[0], Uk_[1], Uk_[2], mDiag);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };
        DeviceBuffer<scalar> r0IC,r0BC,r0lIC,r0lBC;
        deviceBCDivCoeffs(dbU_.comp[0], phiBnd_, r0IC, r0BC);
        deviceBCLaplacianCoeffsFace(dbU_.comp[0], nuEffBnd, r0lIC, r0lBC);
        deviceAxpy(-1.0,r0lIC,r0IC);
        // slip/symmetry: the boundary diagonal differs per component (vf_k=|n_k|), so the SHARED relax/rAU diagonal
        // must use OF's cmptMax(cmptMag(iC0,iC1,iC2)), not comp[0] alone (== comp[0] when components are equal, so
        // every other BC is unaffected). Compute the per-component boundary iC and the per-face max magnitude.
        DeviceBuffer<scalar> iCmaxMag;
        if (hasSym_)
        {
            DeviceBuffer<scalar> r1IC,r1BC,r1lIC,r1lBC, r2IC,r2BC,r2lIC,r2lBC;
            deviceBCDivCoeffs(dbU_.comp[1], phiBnd_, r1IC, r1BC);
            deviceBCLaplacianCoeffsFace(dbU_.comp[1], nuEffBnd, r1lIC, r1lBC);
            deviceAxpy(-1.0,r1lIC,r1IC);
            deviceBCDivCoeffs(dbU_.comp[2], phiBnd_, r2IC, r2BC);
            deviceBCLaplacianCoeffsFace(dbU_.comp[2], nuEffBnd, r2lIC, r2lBC);
            deviceAxpy(-1.0,r2lIC,r2IC);
            deviceCmptMaxMag3(r0IC, r1IC, r2IC, iCmaxMag);   // OF fvMatrix::relax adds cmptMax(cmptMag(iC)) to the diagonal
        }
        // + implicit fvm::ddt(U) DIAGONAL (rho=1 incompressible): coefft*rDeltaT*V, shared by all 3 components, added
        // to the assembled diagonal BEFORE relax -- OF assembles fvm::ddt into UEqn, then UEqn.relax(). No-op for SIMPLE.
        deviceFvmDdtDiag(dm.V, ddtc, 1.0, mDiag);
        DeviceBuffer<scalar> delta;   // mDiagR is now a member
        deviceRelaxDiag(deviceLduView(dm,mDiag,mUp,mLo), dm, r0IC, ctl_.relaxU, mDiagR, delta,
                        hasCyclic_ ? cycSumOff.data() : (hasAMI_ ? amiSumOff.data() : nullptr),
                        hasSym_ ? iCmaxMag.data() : nullptr);
        // velocityDampingConstraint: OF fvOptions.constrain(UEqn) runs AFTER UEqn.relax() and adds an implicit diagonal
        // sink (no source) where |U|>UMax. Add to the relaxed diagonal so it feeds the predictor AND rAU/HbyA (= OF A()).
        if (vdcActive_) deviceFvoVelocityDamping(vdcCells_, vdcUMax_, vdcC_, dm.V, Uk_[0], Uk_[1], Uk_[2], mDiagR);
        const DeviceLduView Uview = deviceLduView(dm, mDiagR, mUp, mLo);
        // rotational cyclic: snapshot U_old so the deferred rotation MIXING is built from ONE consistent field for
        // all three components. OF evaluates the cyclic patchNeighbourField (forwardT.U[nbr]) ONCE at UEqn assembly,
        // from U_old, for the whole vector. cf solves the 3 components SEQUENTIALLY in place (Uk_[kk] is the solution
        // vector), so reading the live Uk_ inside the loop would feed component kk the freshly-solved earlier
        // components -> the x<->y rotation mixing becomes asymmetric -> spurious cross-component coupling that breaks
        // the rotational (axi)symmetry and drifts the converged field. The snapshot makes it order-independent.
        DeviceBuffer<scalar> Usnap[3];
        if ((hasCyclic_ && cyc_.rotational) || (hasAMI_ && ami_.rotational))
            for (int kk = 0; kk < 3; ++kk)
                deviceCopy(Usnap[kk], Uk_[kk]);
        // rotational AMI deferred: the UN-rotated AMI-interpolated neighbour of the U_old snapshot (per src face),
        // consistent across components (the deferred mixing reads it, not the live Uk_).
        DeviceBuffer<scalar> Uint[3];
        if (hasAMI_ && ami_.rotational)
            for (int l = 0; l < 3; ++l)
                deviceAmiInterpolate(ami_, Usnap[l], Uint[l]);
        // linearUpwind / non-orth need grad(U). Precompute ALL 3 component gradients HERE (Uk_ is still all-old at this
        // point, none solved yet), cyclic-inclusive: the cyclic linearUpwind reconstruction rotates gradU[nbr], so it
        // needs every component from a consistent field (computing per-kk inside the loop would read freshly-solved
        // earlier components -> the same sequencing leak as the deferred mixing).
        DeviceBuffer<scalar> gUx[3], gUy[3], gUz[3];
        DeviceBuffer<scalar> amiURot[3];   // rotated AMI-interp of the CURRENT U (gradient face value + linearUpwind/non-orth nbr)
        if ((ctl_.linearUpwind || ctl_.lust || ctl_.nonOrth) && hasAMI_ && ami_.rotational)
            deviceAmiInterpolateVec(ami_, Uk_[0], Uk_[1], Uk_[2], amiURot[0], amiURot[1], amiURot[2]);
        if (ctl_.linearUpwind || ctl_.lust || ctl_.nonOrth)
            for (int l = 0; l < 3; ++l)
            {
                DeviceBuffer<scalar> ubv;
                deviceBCValue(dbU_.comp[l], Uk_[l], ubv);
                deviceGaussGrad(dm, Uk_[l], ubv, gUx[l], gUy[l], gUz[l]);
                if (hasCyclic_)
                {
                    if (cyc_.rotational) deviceCyclicAddGradRot(cyc_, Uk_[0], Uk_[1], Uk_[2], l, dm.V, gUx[l], gUy[l], gUz[l]);
                    else interfaceAddGrad(cyc_, Uk_[l], dm.V, gUx[l], gUy[l], gUz[l]);
                }
                if (hasAMI_)
                {
                    if (ami_.rotational) deviceAmiAddGradRot(ami_, Uk_[l], amiURot[l], dm.V, gUx[l], gUy[l], gUz[l]);
                    else interfaceAddGrad(ami_, Uk_[l], dm.V, gUx[l], gUy[l], gUz[l]);
                }
                // grad(U) cellLimited Gauss linear <k>: clamp this component's gradient so the linearUpwind/non-orth
                // reconstruction stays bounded, OF's primary stabiliser on skewed (snappy) meshes. (min/max gather is
                // over internal + non-cyclic boundary neighbours; interface neighbours not included, fine off-interface.)
                if (ctl_.gradULimitK > 0.0) deviceCellLimitGrad(dm, Uk_[l], ubv, gUx[l], gUy[l], gUz[l], ctl_.gradULimitK);
            }
        // actuationDiskSource (Froude): T = 2*rho*A*(Uref.diskDir)^2*a*(1-a), Uref = mean U over the monitor cells
        // (lagged). Computed once per iter; distributed over the disk cells (V[c]/Vtot) into relaxSrc below.
        scalar adT = 0;
        if (adActive_)
        {
            const scalar ud = (deviceDot(Uk_[0], adMonMask01_)*adDiskDir_.x + deviceDot(Uk_[1], adMonMask01_)*adDiskDir_.y
                             + deviceDot(Uk_[2], adMonMask01_)*adDiskDir_.z) / adNmon_;   // Uref . diskDir
            adT = 2.0 * adArea_ * ud*ud * adA_ * (1.0 - adA_);
        }
        // rotorDiskSource (BEM): per-cell body force from the current U (lagged), recomputed each iter.
        if (rotor_.active) deviceRotorForce(rotor_, nC_, Uk_[0], Uk_[1], Uk_[2], rotorFx_, rotorFy_, rotorFz_);
        DeviceBuffer<scalar>* rotorF[3] = { &rotorFx_, &rotorFy_, &rotorFz_ };
        // linearUpwindV: the V-limiter couples the 3 velocity components, so compute all 3 correction sources once
        // here (the plain per-component linearUpwind path stays inside the kk loop below).
        DeviceBuffer<scalar> corrV[3];
        if (ctl_.linearUpwindV) deviceLinearUpwindVCorr(dm, phiInt_, gUx, gUy, gUz, Uk_[0], Uk_[1], Uk_[2], corrV[0], corrV[1], corrV[2]);
        for (int kk = 0; kk < 3; ++kk)   // (#5 OpenMP-parallel momentum was tried + benchmarked -> 136x SLOWER, reverted)
        {
            DeviceBuffer<scalar> dIC,dBC,lIC,lBC;
            deviceBCDivCoeffs(dbU_.comp[kk],phiBnd_,dIC,dBC);
            deviceBCLaplacianCoeffsFace(dbU_.comp[kk],nuEffBnd,lIC,lBC);
            deviceAxpy(-1.0,lIC,dIC);
            deviceAxpy(-1.0,lBC,dBC);
            iC[kk]=std::move(dIC);
            bCb[kk]=std::move(dBC);
            deviceHadamard(relaxSrc[kk], delta, Uk_[kk]);
            deviceAxpy(1.0, *ddr[kk], relaxSrc[kk]);                                  // += explicit divDevReff stress
            // + fvm::ddt(U) SOURCE for this component: rDeltaT*V*(coefft0*Uold - coefft00*Uold2). Into relaxSrc so it
            // feeds BOTH the predictor solve AND H()/HbyA (OF's UEqn.H() includes the ddt source). No-op for SIMPLE.
            deviceFvmDdtSource(dm.V, ddtc, 1.0, Uold_[kk], Uold2_[kk], relaxSrc[kk], ddtc.cn ? &Uddt0_[kk] : nullptr);   // CrankNicolson: + ocCoeff*ddt0
            if (mrf_.active) deviceMrfCoriolis(mrf_, dm.V, Uk_[0], Uk_[1], Uk_[2], kk, relaxSrc[kk]);   // MRF Coriolis: -V*(Omega x U)_kk on zone cells
            // Explicit momentum corrections (linearUpwind deferred + non-orth laplacian) go into relaxSrc so they
            // feed BOTH the predictor solve AND H()/HbyA, OF's UEqn.H() includes them. Adding them only to the
            // predictor source leaves the corrector inconsistent and the U residual plateaus (does NOT reach 1e-6).
            if (ctl_.linearUpwind || ctl_.lust)   // deferred correction: source -= div(phi * grad(U_kk)_upwind.(Cf-C_up))
            {
                DeviceBuffer<scalar> corr;
                if (ctl_.linearUpwindV) corr = std::move(corrV[kk]);                          // vector-limited (computed once above)
                else                    deviceLinearUpwindCorr(dm, phiInt_, gUx[kk], gUy[kk], gUz[kk], corr);
                if (hasCyclic_) interfaceAddLinUpwindCorr(cyc_, kk, gUx, gUy, gUz, corr);   // + cyclic-face linearUpwind (rotated nbr)
                if (hasAMI_)    interfaceAddLinUpwindCorr(ami_, kk, gUx, gUy, gUz, corr);      // + AMI-face linearUpwind (rotated nbr stencil)
                if (ctl_.lust)   // OF LUST.H = 0.75*linear + 0.25*linearUpwind: scale the lU corr 0.25, add 0.75*linear corr
                {
                    deviceScale(corr, 0.25);
                    DeviceBuffer<scalar> lc;
                    deviceLinearCorr(dm, phiInt_, Uk_[kk], lc);
                    deviceAxpy(0.75, lc, corr);   // (cyclic linear part omitted; LUST cases are non-cyclic)
                }
                deviceAxpy(-1.0, corr, relaxSrc[kk]);
            }
            if (ctl_.nonOrth)   // -fvm::laplacian source correction: -= lapCorr(nuEff, grad(U_kk))
            {
                DeviceBuffer<scalar> lc;
                deviceLaplacianCorr(dm, nuEff_f, gUx[kk], gUy[kk], gUz[kk], lc);
                if (hasCyclic_) interfaceAddLapCorr(cyc_, kk, nuEff, gUx, gUy, gUz, lc);   // + cyclic-face non-orth (rotated tensor)
                if (hasAMI_)    interfaceAddLapCorr(ami_, kk, nuEff, gUx, gUy, gUz, lc);       // + AMI-face non-orth (rotated tensor stencil)
                deviceAxpy(-nonOrthRelaxU, lc, relaxSrc[kk]);
            }
            // constant body force (drives periodic channels) is an explicit momentum SOURCE: it must go into relaxSrc
            // so it feeds BOTH the predictor AND H()/HbyA (OF UEqn.H() includes fvOptions sources). Adding it only to
            // the predictor leaves HbyA short by rAU*g -> the corrector U = HbyA - rAU*grad(p) converges ~(1-relax) low.
            const scalar bf = (kk==0)?ctl_.bodyForce.x:(kk==1)?ctl_.bodyForce.y:ctl_.bodyForce.z;
            if (bf != 0.0) deviceAxpy(bf, dm.V, relaxSrc[kk]);   // += V*g
            if (hasFvoMom_) deviceAxpy(1.0, fvoMomSu_[kk], relaxSrc[kk]);   // == fvOptions(U): explicit momentum source Su*V
            if (por_.active) deviceFvoPorositySource(por_, kk, ctl_.nu, dm.V, Uk_[0], Uk_[1], Uk_[2], relaxSrc[kk]);  // porosity anisotropic remainder
            if (mvfActive_) deviceAxpy((kk==0?mvfFlowDir_.x:kk==1?mvfFlowDir_.y:mvfFlowDir_.z)*mvfGradP_, mvfMaskV_, relaxSrc[kk]);  // meanVelocityForce body force (flowDir*gradP*V)
            // actuationDisk: OF adds T*diskDir DIRECTLY to eqn.source() (= eqn -= ., opposite cf's relaxSrc convention,
            // which mirrors meanVelocityForce's eqn += Su) -> relaxSrc -= (V/Vtot)*T*diskDir (a momentum sink/turbine).
            if (adActive_) deviceAxpy(-adT*(kk==0?adDiskDir_.x:kk==1?adDiskDir_.y:adDiskDir_.z)/adVtot_, adMaskVDisk_, relaxSrc[kk]);
            if (rotor_.active) deviceAxpy(-1.0, *rotorF[kk], relaxSrc[kk]);   // rotorDiskSource: relaxSrc -= force (OF eqn -= force)
            DeviceBuffer<scalar> s;
            deviceHadamard(s, dm.V, *gg[kk]);
            deviceScale(s,-1.0);
            deviceAxpy(1.0,relaxSrc[kk],s);   // s = -V*grad(p) + relaxSrc (incl. body force)
            // rotational cyclic deferred split (PREDICTOR only): the diag-implicit off-diagonal (ifCoeffC = ifCoeff.
            // forwardT[kk][kk]) plus this off-diagonal MIXING term reconstructs the full-rotation coupling -ifCoeff.
            // (forwardT.U[nbr])[kk] in the predictor RHS, but with a consistent (diagonal-dominant) matrix that
            // converges where OF's full-implicit limit-cycles in cf's segregated SIMPLE. Built from the U_old snapshot
            // (order-independent). H() below uses the FULL rotation on the post-solve U (OF-faithful), NOT this term.
            if (hasCyclic_ && cyc_.rotational) interfaceAddDeferredRot(cyc_, Usnap[0], Usnap[1], Usnap[2], kk, s);
            if (hasAMI_ && ami_.rotational)    interfaceAddDeferredRot(ami_, Uint[0], Uint[1], Uint[2], kk, s);
            DeviceBuffer<scalar> diagC,b;
            deviceFold(dm, mDiagR, s, iC[kk], bCb[kk], diagC, b);
            // rotational: the implicit off-diagonal for component kk is scaled by forwardT[kk][kk] (OF transformCoupleField).
            const scalar* mIfc = !hasCyclic_ ? nullptr : (cyc_.rotational ? cyc_.ifCoeffC[kk].data() : cyc_.ifCoeff.data());
            const DeviceLduView mv = hasCyclic_
                ? deviceLduViewCyclic(dm,diagC,mUp,mLo, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), mIfc)
                : hasAMI_
                ? deviceLduViewAmi(dm,diagC,mUp,mLo, ami_.n, ami_.ownCell.data(), ami_.off.data(), ami_.nbrCell.data(), ami_.weight.data(),
                                   ami_.rotational ? ami_.ifCoeffC[kk].data() : ami_.ifCoeff.data())
                : deviceLduView(dm,diagC,mUp,mLo);
            const scalar nf = deviceNormFactor(mv, Uk_[kk], b, ones_);          // OF residualControl normalisation
            // OF fvVectorMatrix::solveSegregated solves each U component with the `U` lduMatrix solver (smoothSolver
            // /GaussSeidel for motorBike). Route through deviceSymGaussSeidel when fvSolution asks for it (robust on
            // anisotropic snappy cells where Jacobi-BiCGStab under-solves the loose relTol); interface LDUs keep BiCGStab.
            scalar ur;
            DeviceSolverPerf uperf;
            if (ctl_.gsU && !hasCyclic_ && !hasAMI_)
                ur = deviceSymGaussSeidel(mv, b, Uk_[kk], nf, tol, ctl_.relTolU, 5000, &uperf);
            else
            {
                uperf = deviceJacobiBiCGStab(mv, b, Uk_[kk], nf, tol, ctl_.relTolU, 5000, ctl_.bicgCheckEvery);
                ur = uperf.initialResidual;
            }
            // keep all 3 components + their final residual / nIter (OF prints Solving for Ux/Uy/Uz each iteration)
            if (kk == 0)      { res.Ux = ur; res.UxFinal = uperf.finalResidual; res.UxIters = uperf.nIterations; }
            else if (kk == 1) { res.Uy = ur; res.UyFinal = uperf.finalResidual; res.UyIters = uperf.nIterations; }
            else              { res.Uz = ur; res.UzFinal = uperf.finalResidual; res.UzIters = uperf.nIterations; }
        }
    }

    void DeviceSimpleSolver::correctPressureVelocity(DeviceSimpleResidual& res)
    {
        const DeviceMesh& dm = dm_;
        // Non-orth pressure-correction limiter (OF fv::limitedSnGrad), read by the pressure phase below: per-face caps the
        // lagged corrVec.grad(p) correction to (psi/(1-psi))*|orthogonal snGrad| on pathological high-non-orth meshes
        // (T3A: AR 3232, 43.8deg), where the once-lagged correction outruns the over-relaxed diagonal and diverges. Only
        // the pathological faces (|corr|>|orth|) are limited; well-behaved faces keep limiter==1 (full "corrected",
        // OF-accurate). psi=1.0 -> unlimited (bit-identical to the validated cases). The momentum-viscous correction is
        // stable at full strength (it is the PRESSURE correction that destabilizes on T3A), so the predictor keeps psi=1.
        const scalar nonOrthLimitP = std::getenv("BRAE_NONORTH_LIMIT") ? std::atof(std::getenv("BRAE_NONORTH_LIMIT")) : ctl_.nonOrthLimit;
        // The predictor's LDU view + grad-pointer array, rebuilt over the (now member) relaxed-matrix + gradient buffers
        // for the pressure-velocity phase below (H() at deviceMatrixH, and the HbyA non-orth grad term). The predictor
        // does not modify mUp/mLo/gx/gy/gz after filling them, so these are identical pointers -> identical results.
        const DeviceLduView Uview = deviceLduView(dm, mDiagR, mUp, mLo);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };
        // OF A() = (relaxed_diag + cmptAv(internalCoeffs))/V  (fvMatrix::D() -> addCmptAvBoundaryDiag, fvMatrix::A()).
        // For component-INDEPENDENT BCs cmptAv(iC) == iC[0] (bit-identical to before). For slip/symmetry the boundary
        // diagonal is PER-COMPONENT (vf_k=|n_k|), so the SHARED rAU must use the component-AVERAGE, NOT iC[0]: iC[0]
        // misses the constrained-component diagonal (bottom symmetry iC=(0,iC_v,0): iC[0]=0 but cmptAv=iC_v/3), which
        // on high-AR cells made rAU wildly too large -> catastrophic slip blowup. Reused below as the H() cmptAv term.
        DeviceBuffer<scalar> cmptAvIC;
        if (hasSym_)
        {
            deviceCopy(cmptAvIC, iC[0]);
            deviceAxpy(1.0, iC[1], cmptAvIC);
            deviceAxpy(1.0, iC[2], cmptAvIC);
            deviceScale(cmptAvIC, 1.0/3.0);
        }
        DeviceBuffer<scalar> diagA,dumb,rAU;
        deviceFold(dm,mDiagR,zeroSrc_, hasSym_ ? cmptAvIC : iC[0], zeroBndU_,diagA,dumb);
        deviceReciprocalV(dm,diagA,rAU);
        // SIMPLEC: rAtU = 1/(1/rAU - H1) = V/max(A*1, 0.1*diagA) (A*1 = row sum = deviceAmul with ones). drAtU = rAtU - rAU.
        DeviceBuffer<scalar> rAtU, drAtU;
        if (ctl_.consistent)
        {
            DeviceBuffer<scalar> rowSum;
            deviceAmul(hasCyclic_
                ? deviceLduViewCyclic(dm, diagA, mUp, mLo, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), cyc_.ifCoeff.data())
                : deviceLduView(dm, diagA, mUp, mLo), ones_, rowSum);
            deviceSimplecRAtU(dm, rowSum, diagA, rAtU);
            deviceCopy(drAtU, rAtU);
            deviceAxpy(-1.0, rAU, drAtU);
            // OF rhoSimpleFoam/pcEqn.H weights the SIMPLEC flux correction by density:
            //     phiHbyA += fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*magSf
            // (incompressible simpleFoam/pEqn.H has no rho and is the expression below without this).
            // The HbyA correction on the other hand is NOT weighted in either: HbyA -= (rAU - rAtU)*grad(p),
            // so only drAtU used for the FLUX is scaled -- keep an unscaled copy for that.
            if (compressible_)
            {
                DeviceBuffer<scalar> t;
                deviceHadamard(t, th_.rho, drAtU);
                deviceCopy(drAtUFlux_, t);
            }
            else deviceCopy(drAtUFlux_, drAtU);
        }
        else
            deviceCopy(rAtU, rAU);
        // meanVelocityForce.correct (after the predictor, before H()/HbyA, like OF fvOptions.correct(U) in UEqn.H):
        // dGradP = relax*(|Ubar| - magUbarAve)/rAUave; accumulate gradP_; correct U += flowDir*rAU*dGradP. Reductions
        // and the body force/correction use maskV/mask01 so selectionMode all (whole field) and cellZone both work.
        if (mvfActive_)
        {
            const scalar magUbarAve = (mvfFlowDir_.x*deviceDot(Uk_[0], mvfMaskV_) + mvfFlowDir_.y*deviceDot(Uk_[1], mvfMaskV_)
                                     + mvfFlowDir_.z*deviceDot(Uk_[2], mvfMaskV_)) / mvfVtot_;
            const scalar rAUave = deviceDot(rAU, mvfMaskV_) / mvfVtot_;
            const scalar dGradP = mvfRelax_ * (mvfUbarMag_ - magUbarAve) / rAUave;
            mvfGradP_ += dGradP;
            if (std::getenv("BRAE_MVF_DEBUG")) std::printf("    mvf: magUbarAve=%.6g rAUave=%.6g dGradP=%.6g gradP=%.6g\n", magUbarAve, rAUave, dGradP, mvfGradP_);
            DeviceBuffer<scalar> rAUm;
            deviceHadamard(rAUm, rAU, mvfMask01_);              // rAU in the selection, 0 else
            deviceAxpy(mvfFlowDir_.x*dGradP, rAUm, Uk_[0]);
            deviceAxpy(mvfFlowDir_.y*dGradP, rAUm, Uk_[1]);
            deviceAxpy(mvfFlowDir_.z*dGradP, rAUm, Uk_[2]);
        }
        if (limUActive_) deviceFvoLimitVelocity(limUCells_, limUMax_, Uk_[0], Uk_[1], Uk_[2]);   // limitVelocity on the predictor U (OF fvOptions.correct) -> feeds H()/HbyA
        // slip/symmetry H() consistency term: OF UEqn.H() adds (cmptAv(iC) - iC_cmpt)*psi_cmpt per component, where
        // cmptAv = mean of the 3 boundary internalCoeffs (fvMatrix::H + addCmptAvBoundaryDiag). Zero for every
        // component-independent BC (iC_cmpt == cmptAv), nonzero ONLY for slip (per-component vf=|n_k|), it supplies
        // the HbyA<->rAU consistency that a single scalar rAU otherwise breaks at a non-axis-aligned wall.
        DeviceBuffer<scalar>& cmptAvH = cmptAvIC;   // same cmptAv(iC) already computed for the rAU diagonal above
        DeviceBuffer<scalar> HbyA[3];
        {
            DeviceBuffer<scalar> Hk[3];
            // cyclicAMI H: precompute the AMI-interpolated (rotated) neighbour U once for the vector, then H[own] -=
            // ifCoeff*UNbr[kk]/V per component (= OF UEqn.H() with the AMI weighted stencil).
            DeviceBuffer<scalar> UNx, UNy, UNz;
            if (hasAMI_) deviceAmiInterpolateVec(ami_, Uk_[0], Uk_[1], Uk_[2], UNx, UNy, UNz);
            DeviceBuffer<scalar>* UN[3] = { &UNx, &UNy, &UNz };
            for (int kk = 0; kk < 3; ++kk)
            {
                DeviceBuffer<scalar> bdH;   // slip: bdDiag = cmptAv(iC) - iC[kk] (OF H() term); else zero (bit-identical)
                if (hasSym_)
                {
                    deviceCopy(bdH, cmptAvH);
                    deviceAxpy(-1.0, iC[kk], bdH);
                }
                deviceMatrixH(Uview, dm, Uk_[kk], relaxSrc[kk], hasSym_ ? bdH : zeroBndU_, bCb[kk], Hk[kk]);
                if (hasCyclic_ && !cyc_.rotational) interfaceAddH(cyc_, Uk_[kk], dm.V, Hk[kk]);   // translational off-diag
                if (hasAMI_) interfaceAddH(ami_, *UN[kk], dm.V, Hk[kk]);
            }
            // rotational cyclic H: the FULL rotation -ifCoeff.(forwardT.U[nbr])[kk] on the POST-solve U, evaluated once
            // for the whole vector (matches OF UEqn.H(), which re-evaluates patchNeighbourField from the new U). The
            // deferred mixing fed only the PREDICTOR; H is the exact full-rotation explicit coupling, consistent across
            // components, so HbyA/phiHbyA carry the true rotated neighbour momentum, not a per-component-stale value.
            if (hasCyclic_ && cyc_.rotational) deviceCyclicAddHRot(cyc_, Uk_[0], Uk_[1], Uk_[2], dm.V, Hk[0], Hk[1], Hk[2]);
            for (int kk = 0; kk < 3; ++kk)
                deviceHadamard(HbyA[kk], rAU, Hk[kk]);
        }
        DeviceBuffer<scalar> phiHi;
        deviceVectorFlux(dm, HbyA[0], HbyA[1], HbyA[2], phiHi);   // flux of the ORIGINAL HbyA
        if (hasCyclic_)
        {
            if (cyc_.rotational) deviceCyclicFluxRot(cyc_, HbyA[0], HbyA[1], HbyA[2]);   // rotate the neighbour HbyA
            else interfaceFlux(cyc_, HbyA[0], HbyA[1], HbyA[2]);   // cyclic-face phiHbyA = interp(HbyA).Sf -> cyc_.phi
        }
        if (hasAMI_) interfaceFlux(ami_, HbyA[0], HbyA[1], HbyA[2]);   // AMI-face phiHbyA = (w*HbyA[own]+(1-w)*interp(HbyA[nbr])).Sf
        DeviceBuffer<scalar> hxb,hyb,hzb;
        deviceBCValue(dbU_.comp[0],HbyA[0],hxb);
        deviceBCValue(dbU_.comp[1],HbyA[1],hyb);
        deviceBCValue(dbU_.comp[2],HbyA[2],hzb);
        // constrainHbyA at mixed velocity faces (OF resets phiHbyA_b = U_b.Sf at fixesValue patches): use U_b not HbyA_b.
        if (hasMixed_) deviceConstrainMixedHbyA(dbU_, Uk_[0], Uk_[1], Uk_[2], hxb, hyb, hzb);
        // slip/symmetry: OF constrainHbyA sets HbyA_b = U_b (= U.boundaryField, the slipped velocity) at every
        // non-assignable patch, NOT the projected HbyA. Project the cell U (HbyA_b = U_c - n(n.U_c) = U_b), matching
        // OF byte-for-byte and the mixed path above (both use U_b). Wall flux is still exactly 0 (normal removed).
        if (hasSym_) deviceConstrainSymmetryHbyA(dbU_, Uk_[0], Uk_[1], Uk_[2], hxb, hyb, hzb);
        DeviceBuffer<scalar> phiHb;
        deviceBoundaryFlux(dm,hxb,hyb,hzb,phiHb);
        if (compressible_)
        {
            // OF rhoSimpleFoam pEqn.H: phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA), i.e. a MASS flux.
            // Both halves must be weighted: deviceInterpolate covers internal faces only, and the boundary
            // rho comes from the boundary p and T rather than the adjacent cell -- see
            // deviceThermoRhoBoundary for why extrapolating there is silently wrong at a fixed-T inlet.
            DeviceBuffer<scalar> rhoF;
            deviceInterpolate(dm, th_.rho, rhoF);
            DeviceBuffer<scalar> tmp;
            deviceHadamard(tmp, rhoF, phiHi);
            deviceCopy(phiHi, tmp);

            DeviceBuffer<scalar> rhoB;
            deviceThermoRhoBoundary(dbP_, dp_, dbHe_, th_.he, tc_, rhoB);
            DeviceBuffer<scalar> tmpB;
            deviceHadamard(tmpB, rhoB, phiHb);
            deviceCopy(phiHb, tmpB);
        }
        if (mrf_.active) deviceMrfApplyFrameFlux(mrf_, +1.0, phiHi, phiHb);   // MRF.makeRelative(phiHbyA) before pressure
        // adjustPhi (OF order: after flux(HbyA), before the SIMPLEC correction): enforce global continuity by
        // scaling the adjustable outflow when there is no pressure reference (closed/all-velocity domains).
        if (ctl_.needRef) deviceAdjustPhi(adjustMask_, phiHb);
        if (ctl_.consistent)
        {
            // phiHbyA += interpolate(rAtU-rAU)*snGrad(p)*magSf = laplacian(interp(drAtU), p).flux()  (NOT via HbyA flux)
            DeviceBuffer<scalar> drAtUf;
            deviceInterpolate(dm, drAtUFlux_, drAtUf);   // rho*(rAtU-rAU) when compressible
            DeviceBuffer<scalar> ld, lu, ll;
            deviceLaplacianCoeffs(dm, drAtUf, ld, lu, ll, ctl_.nonOrth);   // over-relaxed nonOrthDeltaCoeffs to match OF's corrected snGrad(p) (was orthogonal dc -> cos(theta) too small on non-orth meshes)
            DeviceBuffer<scalar> fInt;
            deviceMatrixFluxInternal(deviceLduView(dm, ld, lu, ll), dp_, fInt);
            deviceAxpy(1.0, fInt, phiHi);
            // OF's term is interp(rAtU-rAU)*fvc::snGrad(p)*magSf, and snGrad(p) is the CORRECTED snGrad (orth + non-orth).
            // The orth part is the laplacian flux above; add the non-orth part interp(drAtU)*(corrVec.grad(p)_f)*magSf so
            // the SIMPLEC flux correction is complete on non-orthogonal meshes (no-op when orthogonal: corrVec~=0).
            if (ctl_.nonOrth)
            {
                DeviceBuffer<scalar> ffcS;
                deviceLaplacianCorrFlux(dm, drAtUf, gx, gy, gz, ffcS);
                deviceAxpy(1.0, ffcS, phiHi);
            }
            DeviceBuffer<scalar> dIC, dBC;
            deviceBCLaplacianCoeffs(dbP_, drAtUFlux_, dIC, dBC);
            DeviceBuffer<scalar> fBnd;
            deviceMatrixFluxBoundary(dbP_, dIC, dBC, dp_, fBnd);
            deviceAxpy(1.0, fBnd, phiHb);
            // HbyA adjustment AFTER the flux (feeds the velocity corrector only): HbyA -= (rAU-rAtU)*grad(p)
            for (int kk = 0; kk < 3; ++kk)
            {
                DeviceBuffer<scalar> t;
                deviceHadamard(t, drAtU, *gg[kk]);
                deviceAxpy(1.0, t, HbyA[kk]);
            }
        }
        DeviceBuffer<scalar> rAUf;
        if (compressible_)
        {
            // OF rhoSimpleFoam pEqn.H: rhorAUf = fvc::interpolate(rho*rAU). The laplacian coefficient is
            // the ONLY change the subsonic pressure equation needs -- there is no psi*p term outside the
            // transonic branch, so the system stays symmetric and the AMG-PCG path is unchanged.
            DeviceBuffer<scalar> rhoRAtU;
            deviceHadamard(rhoRAtU, th_.rho, rAtU);
            deviceInterpolate(dm, rhoRAtU, rAUf);
        }
        else
        {
            deviceInterpolate(dm,rAtU,rAUf);
        }
        // pressure matrix buffers are PERSISTENT members: their addresses stay fixed across SIMPLE steps (only the
        // values change), so the V-cycle CUDA graph (keyed on diagCp_) is captured once and replayed every step.
        DeviceBuffer<scalar> pPrev;
        deviceCopy(pPrev, dp_);   // p entering the pressure step (for relaxation), captured once
        // nNonOrthogonalCorrectors (OF SIMPLE, pEqn.H `while (simple.correctNonOrthogonal())`): re-solve
        // laplacian(rAtU,p) == div(phiHbyA) (nNonOrth+1) times, recomputing the EXPLICIT non-orthogonal correction
        // (corrVec.grad(p)) from the UPDATED p each pass; the conservative flux is committed only on the final pass.
        // Orthogonal scheme -> corrVec = 0 so the correction vanishes and one pass suffices (identical to nNonOrth=0).
        // The pEqn is fully re-assembled each pass, exactly as OF rebuilds the fvMatrix (deviceFoldPressure WRITES
        // diagCp_/bp_, and setReference doubles a FRESH diag, so per-pass re-assembly is self-consistent). Unchanged
        // for SIMPLE vs SIMPLEC: rAtU is just the diffusivity (= rAU for SIMPLE, = 1/(1/rAU - H1) for SIMPLEC); the
        // one-time SIMPLEC phiHbyA/HbyA correction above already used the entry p, as in OF (it is outside this loop).
        const int nPass = (ctl_.nonOrth ? ctl_.nNonOrth : 0) + 1;
        DeviceBuffer<scalar> pIC, pBC, ffcP, ffcPcyc, ffcPami;
        DeviceSolverPerf pp, pp0;
        for (int nonOrthPass = 0; nonOrthPass < nPass; ++nonOrthPass)
        {
            if (nonOrthPass > 0)   // recompute grad(p) from the just-solved p (pass 0 uses the entry grad in gx/gy/gz)
            {
                DeviceBuffer<scalar> pbvN;
                deviceBCValue(dbP_, dp_, pbvN);
                deviceGaussGrad(dm, dp_, pbvN, gx, gy, gz);
                if (hasCyclic_) interfaceAddGrad(cyc_, dp_, dm.V, gx, gy, gz);
                if (hasAMI_)    interfaceAddGrad(ami_, dp_, dm.V, gx, gy, gz);
            }
            deviceLaplacianCoeffs(dm, rAUf, pD_, pU_, pL_, ctl_.nonOrth);
            deviceBCLaplacianCoeffs(dbP_, rAtU, pIC, pBC);
            DeviceBuffer<scalar> divPhiH;
            deviceDiv(dm, phiHi, phiHb, divPhiH);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, divPhiH);            // + cyclic-face flux into continuity div(phiHbyA)
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, divPhiH);               // + AMI-face flux into continuity div(phiHbyA)
            deviceFoldPressure(dm, pD_, divPhiH, pIC, pBC, diagCp_, bp_);
            // cyclic pressure Laplacian laplacian(rAtU,p): fold the interface diagonal into diagCp_ + set cyc_.ifCoeff.
            if (hasCyclic_) interfaceAssembleLaplacian(cyc_, rAtU, diagCp_, /*addToDiag*/true);
            if (hasAMI_)    interfaceAssembleLaplacian(ami_, rAtU, diagCp_, /*addToDiag*/true);   // AMI pressure laplacian
            // explicit non-orth correction: faceFluxCorr from grad(p) for THIS pass; add -V*div(ffc) to b, and subtract
            // ffc from the reconstructed flux below (so div(phi)=0 on non-orth faces). gx/gy/gz = grad(p) this pass.
            ffcP = DeviceBuffer<scalar>();
            ffcPcyc = DeviceBuffer<scalar>();
            ffcPami = DeviceBuffer<scalar>();
            if (ctl_.nonOrth)
            {
                deviceLaplacianCorrFluxLimited(dm, rAUf, dp_, gx, gy, gz, nonOrthLimitP, ffcP);
                DeviceBuffer<scalar> sc;
                deviceFaceDivSource(dm, ffcP, sc);
                deviceAxpy(1.0, sc, bp_);
                if (hasCyclic_) interfaceLapCorrP(cyc_, rAtU, gx, gy, gz, bp_, ffcPcyc);
                if (hasAMI_)    interfaceLapCorrP(ami_, rAtU, gx, gy, gz, bp_, ffcPami);
            }
            // No fixedValue-p patch -> singular all-Neumann pressure. cyclic/AMI use a tiny SYMMETRIC diagonal shift
            // (the periodic RHS is zero-mean); non-cyclic keeps the OF single-cell setReference (doubles a fresh diag).
            if (ctl_.needRef)
            {
                if (hasCyclic_ || hasAMI_)
                {
                    const scalar eps = -1e-10 * (deviceSumMag(diagCp_) / nC_);
                    deviceAxpy(eps, ones_, diagCp_);
                }
                else deviceSetReference(diagCp_, bp_, ctl_.pRefCell, ctl_.pRefValue);
            }
            if (hasCyclic_ || hasAMI_)
            {
                // Periodic/AMI: interface-coupled operator solved with Jacobi-PCG (no AMG; the internal-face Galerkin
                // coarse operator cannot represent the interface edges).
                const DeviceLduView pvc = hasCyclic_
                    ? deviceLduViewCyclic(dm,diagCp_,pU_,pL_, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), cyc_.ifCoeff.data())
                    : deviceLduViewAmi(dm,diagCp_,pU_,pL_, ami_.n, ami_.ownCell.data(), ami_.off.data(), ami_.nbrCell.data(), ami_.weight.data(), ami_.ifCoeff.data());
                const scalar nfp = deviceNormFactor(pvc, dp_, bp_, ones_);
                pp = deviceJacobiPCG(pvc, bp_, dp_, nfp, ctl_.tolP, ctl_.relTolP, 3000);
            }
            else
            {
                const scalar nfp = deviceNormFactor(deviceLduView(dm,diagCp_,pU_,pL_), dp_, bp_, ones_);
                amgGalerkin(amg_, diagCp_, pU_, pL_);                                      // re-coarsen the pressure matrix
                pp = deviceAMGPCG(deviceLduView(dm,diagCp_,pU_,pL_), amg_, bp_, dp_, nfp, ctl_.tolP, ctl_.relTolP, 3000, ctl_.useGraph, ctl_.pcgCheckEvery, ctl_.corrScaling);
            }
            if (nonOrthPass == 0) pp0 = pp;   // SIMPLE residualControl uses the FIRST pass's initial residual (OF convention)
        }
        res.p = pp0.initialResidual;
        res.pFinal = pp.finalResidual;
        res.pIters = pp.nIterations;
        if (std::getenv("BRAE_SOLVER_DEBUG")) std::printf("    p %s iters=%d  init=%.2e final=%.2e\n",
                                                        (hasCyclic_||hasAMI_)?"Jacobi-PCG":"AMG-PCG", pp.nIterations, pp.initialResidual, pp.finalResidual);
        const DeviceLduView pview = deviceLduView(dm,diagCp_,pU_,pL_);
        DeviceBuffer<scalar> pfi;
        deviceMatrixFluxInternal(pview, dp_, pfi);
        deviceCopy(phiInt_, phiHi);
        deviceAxpy(-1.0, pfi, phiInt_);
        if (ctl_.nonOrth) deviceAxpy(-1.0, ffcP, phiInt_);   // phi -= faceFluxCorrection (continuity on non-orth faces)
        DeviceBuffer<scalar> pfb;
        deviceMatrixFluxBoundary(dbP_, pIC, pBC, dp_, pfb);
        deviceCopy(phiBnd_, phiHb);
        deviceAxpy(-1.0, pfb, phiBnd_);
        if (hasCyclic_) interfaceCorrectFlux(cyc_, dp_);   // cyc_.phi -= ifCoeff*(p_nbr - p_own) : conservative periodic flux (pre-relax dp_)
        if (hasAMI_)    interfaceCorrectFlux(ami_, dp_);       // ami_.phi -= ifCoeff*(interp(p_nbr) - p_own)
        if (hasCyclic_ && ctl_.nonOrth && ffcPcyc.size()) deviceAxpy(-1.0, ffcPcyc, cyc_.phi);   // cyclic-face non-orth flux correction (matches phiInt_ -= ffcP)
        if (hasAMI_ && ctl_.nonOrth && ffcPami.size()) deviceAxpy(-1.0, ffcPami, ami_.phi);       // AMI-face non-orth flux correction
        deviceScale(dp_, ctl_.relaxP);
        deviceAxpy(1.0 - ctl_.relaxP, pPrev, dp_);
        DeviceBuffer<scalar> pbv2;
        deviceBCValue(dbP_, dp_, pbv2);
        DeviceBuffer<scalar> gnx,gny,gnz;
        deviceGaussGrad(dm, dp_, pbv2, gnx,gny,gnz);
        if (hasCyclic_) interfaceAddGrad(cyc_, dp_, dm.V, gnx,gny,gnz);   // + cyclic-face contribution to grad(p)
        if (hasAMI_)    interfaceAddGrad(ami_, dp_, dm.V, gnx,gny,gnz);       // + AMI-face contribution to grad(p)
        DeviceBuffer<scalar>* gn[3]={&gnx,&gny,&gnz};
        for (int kk = 0; kk < 3; ++kk)   // SIMPLEC: rAtU
        {
            DeviceBuffer<scalar> Un;
            deviceCorrector(HbyA[kk], rAtU, *gn[kk], Un);
            Uk_[kk]=std::move(Un);
        }
        // limitVelocity (fvOptions.correct): clamp |U| <= max on the corrected (output) velocity. OF clamps after the
        // momentum predictor; cf clamps the post-corrector U so the WRITTEN field is bounded (matches OF's output).
        if (limUActive_) deviceFvoLimitVelocity(limUCells_, limUMax_, Uk_[0], Uk_[1], Uk_[2]);
    }

    DeviceSimpleResidual DeviceSimpleSolver::step()
    {
        const DeviceMesh& dm = dm_;
        DeviceSimpleResidual res;
        solveMomentumPredictor(res);
        correctPressureVelocity(res);

        correctTurbulence();
        // OF continuityErrs.H: contErr = fvc::div(phi) on the CORRECTED flux; sum local = sum(V|contErr|)/sumV,
        // global = sum(V contErr)/sumV. deviceDiv gives Sum(phi)/V per cell, so R = V*div = the cell flux balance.
        {
            DeviceBuffer<scalar> divPhi;
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            DeviceBuffer<scalar> R;
            deviceHadamard(R, divPhi, dm.V);
            const scalar sumV = deviceDot(dm.V, ones_) + 1e-300;
            res.contLocal  = deviceSumMag(R) / sumV;
            res.contGlobal = deviceDot(R, ones_) / sumV;
        }
        return res;
    }

    // Advance one time level: roll the old-time velocity fields (t-2 <- t-1 <- current) and the time steps
    // (deltaT0 <- deltaT). Call ONCE at the top of each time step, before the outer-corrector loop -- OF does this at
    // runTime++ (the registry ages every field's oldTime). First call: Uold2_ stays empty + deltaT0_=0, so ddtCoeffs()
    // bootstraps to Euler exactly like pimpleFoam's first step (only one old level exists yet).
    void DeviceSimpleSolver::advanceTime(scalar deltaT)
    {
        const bool cn  = (ddtScheme_ == DdtScheme::CrankNicolson);
        const bool two = (ddtScheme_ == DdtScheme::backward) || cn;   // backward (coefft00 term) + CrankNicolson (ddt0 recurrence) both need the 2nd old level
        for (int k = 0; k < 3; ++k)
        {
            if (two && Uold_[k].size()) deviceCopy(Uold2_[k], Uold_[k]);   // t-2 <- t-1  (once t-1 exists)
            deviceCopy(Uold_[k], Uk_[k]);                                  // t-1 <- current
        }
        if (ctl_.turbulent && !ctl_.les)   // turbulence old-time levels for fvm::ddt(k/eps): dk_ = k (or nuTilda), de_ = epsilon|omega. Pure LES nut is algebraic -> no ddt.
        {
            if (two && kOld_.size()) deviceCopy(kOld2_, kOld_);
            deviceCopy(kOld_, dk_);
            if (!ctl_.sa)   // SA is one-equation (no second scalar)
            {
                if (two && e2Old_.size()) deviceCopy(e2Old2_, e2Old_);
                deviceCopy(e2Old_, de_);
            }
            if (ctl_.lm)   // kOmegaSSTLM transition fields (ReThetat, gammaInt)
            {
                if (two && ReThetatOld_.size()) deviceCopy(ReThetatOld2_, ReThetatOld_);
                deviceCopy(ReThetatOld_, ReThetat_);
                if (two && gammaIntOld_.size()) deviceCopy(gammaIntOld2_, gammaIntOld_);
                deviceCopy(gammaIntOld_, gammaInt_);
            }
        }
        if (cn)   // CrankNicolson: update each transported variable's stored old ddt (ddt0) from the just-rotated old levels.
        {         // deltaT0 = the PREVIOUS deltaT (deltaT_ not yet overwritten); no-op on the first step (no oldTime.oldTime).
            const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT, deltaT_, ocCoeff_, cnWarm_);
            for (int k = 0; k < 3; ++k) deviceFvmDdtUpdateDdt0(ddtc, Uold_[k], Uold2_[k], Uddt0_[k]);
            if (ctl_.turbulent && !ctl_.les)
            {
                deviceFvmDdtUpdateDdt0(ddtc, kOld_, kOld2_, kddt0_);
                if (!ctl_.sa) deviceFvmDdtUpdateDdt0(ddtc, e2Old_, e2Old2_, e2ddt0_);
                if (ctl_.lm)
                {
                    deviceFvmDdtUpdateDdt0(ddtc, ReThetatOld_, ReThetatOld2_, ReThetatddt0_);
                    deviceFvmDdtUpdateDdt0(ddtc, gammaIntOld_, gammaIntOld2_, gammaIntddt0_);
                }
            }
            if (ddtc.cn && Uold2_[0].size()) cnWarm_ = true;   // first (Euler-startup) ddt0 update done -> subsequent updates use 1+oc
        }
        deltaT0_ = deltaT_;
        deltaT_  = deltaT;
    }

    // One PIMPLE time step -- the transient counterpart of step(): the SAME three composable phases re-orchestrated under
    // outer (momentum<->pressure<->turbulence) + inner (pressure) corrector loops, with fvm::ddt(U) folded into the
    // predictor (active once setDdtScheme() set ddtScheme_ != steadyState). The momentum matrix is re-assembled each
    // outer corrector from the latest U/phi, exactly as OF's while(pimple.loop()) re-forms UEqn.
    void DeviceSimpleSolver::setCompressible(
        const ThermoCoeffs& tc,
        const RhoSimpleControls& rc,
        DeviceBoundary dbHe)
    {
        tc_ = tc;
        rc_ = rc;
        dbHe_ = std::move(dbHe);
        th_.allocate(dm_.nCells);
        compressible_ = true;
    }

    void DeviceSimpleSolver::setAlphatPrt(const std::vector<scalar>& prtFace)
    {
        if (!prtFace.empty()) prtBnd_.copyFrom(prtFace);
    }


    // See the header for why the phases are ordered UEqn -> EEqn -> pEqn -> thermo -> turbulence.
    DeviceSimpleResidual DeviceSimpleSolver::rhoSimpleStep()
    {
        const DeviceMesh& dm = dm_;
        DeviceSimpleResidual res;

        // muEff. The momentum assembly builds nuEff = dnut_ + nuConst_ and nuConst_ has exactly two uses
        // (seeded from ctl_.nu at construction, added here), so handing it the DYNAMIC viscosity turns the
        // same expression into muEff without touching the assembly. mu is T-dependent under Sutherland, so
        // it must be refreshed every outer iteration, not seeded once. dnut_ is zero while the solve is
        // laminar; phase 4 fills it with mut and this line keeps working unchanged.
        deviceCopy(nuConst_, th_.mu);

        // Boundary rho and mu for this iteration. Both are T-dependent, so they move with the solution and
        // must be refreshed here rather than seeded once. nuBndConst_ becomes mu_b, which turns the boundary
        // expression nuEffBnd = nuBndConst_ + nut_b into muEff_b = mu_b + rho_b*nut_b once the wall nut is
        // rho_b-weighted (see addWallNutToMuEff).
        deviceThermoRhoBoundary(dbP_, dp_, dbHe_, th_.he, tc_, rhoBnd_);
        deviceThermoMuBoundary(dbHe_, th_.he, tc_, muBnd_);
        deviceThermoNuBoundary(dbP_, dp_, dbHe_, th_.he, tc_, nuWallBnd_);   // OF nu(patchi), for the wall functions
        deviceGatherWallNu(wfBndIdx_, nuWallBnd_, wfNu_);   // same nu, in the wall-face ordering omega/G0 uses
        deviceCopy(nuBndConst_, muBnd_);

        // flowRateInletVelocity (massFlowRate): OF recomputes avgU = -mdot/gSum(rho*magSf) every time the
        // U boundary is updated, so it tracks the inlet density as the solution develops. Done here, before
        // the momentum predictor, so the predictor sees the same inlet OF's does.
        for (std::size_t k = 0; k < frPatches_.size(); ++k)
        {
            const scalar sumRhoA = deviceDot(rhoBnd_, frMagSf_[k]);
            if (sumRhoA <= 0.0) continue;
            deviceUpdateFlowRateInlet(dbU_, frMagSf_[k], -frPatches_[k].mdot / sumRhoA, frNx_, frNy_, frNz_);
        }

        solveMomentumPredictor(res);

        // EEqn, with OF's kinetic-energy term: EEqn.H is div(phi,he) + div(phi,K) - laplacian(alphaEff,he).
        // K is built from the velocity the momentum predictor just produced, matching OF's ordering.
        // sensibleInternalEnergy convects Ekp = 0.5|U|^2 + p/rho, sensibleEnthalpy just K = 0.5|U|^2.
        // p_b comes from the pressure BC and rho_b from the boundary p and T, so the boundary half of
        // div(phi,Ekp) is evaluated on the patch exactly as OF's fvc::div does, not extrapolated.
        // div(phi) for the energy equation's "bounded" term. Was passed as zeros, which silently
        // disabled the term even when fvSchemes asked for it.
        DeviceBuffer<scalar> divPhiHe;
        deviceDiv(dm, phiInt_, phiBnd_, divPhiHe);
        DeviceBuffer<scalar> kineticSrc;
        DeviceBuffer<scalar> pBndK;
        if (tc_.internalEnergy) deviceBCValue(dbP_, dp_, pBndK);
        deviceEnergyKineticSource(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], phiInt_, phiBnd_, kineticSrc,
                                  tc_.internalEnergy ? &dp_ : nullptr,
                                  tc_.internalEnergy ? &th_.rho : nullptr,
                                  tc_.internalEnergy ? &pBndK : nullptr,
                                  tc_.internalEnergy ? &rhoBnd_ : nullptr,
                                  &dbHe_, ctl_.limitedHe, ctl_.twoBykHe);   // div(phi,K|Ekp) scheme
        // alphaEff on the PATCHES, so the alphatWallFunction actually reaches the wall heat flux.
        // ALWAYS when compressible, not just when turbulent. OF's laplacian uses the PATCH diffusivity,
        // alpha_b = alphah(p_b, T_b) evaluated at the BOUNDARY temperature. With transport const that
        // equals the adjacent cell value and the distinction is invisible; under sutherland a 700 K wall
        // against a 370 K first cell makes mu_b ~1.6x the cell mu, and using the cell value understates
        // the wall heat flux by that much while converging perfectly.
        DeviceBuffer<scalar> alphaEffBnd;
        if (compressible_)
            deviceAlphaEffBoundary(muBnd_, rhoBnd_,
                                   ctl_.turbulent ? &nutBndAll_ : nullptr,
                                   prtBnd_.size() ? &prtBnd_ : nullptr,
                                   tc_, alphaEffBnd);
        deviceSolveEnergy(
            dm,
            dbHe_,
            th_,
            tc_,
            phiInt_,
            phiBnd_,
            divPhiHe,
            ctl_.boundedHe,
            ctl_.limitedHe,   // div(phi,h|e) limitedLinear, from fvSchemes -- was hardcoded false
            ctl_.luHe,        // div(phi,h|e) linearUpwind,   from fvSchemes -- was hardcoded false
            ctl_.nonOrth,
            ctl_.twoBykHe,
            rc_.relaxHe,
            eTol_,              // fvSolution solvers.(h|e).tolerance -- was hardcoded 1e-10
            eRelTol_,           //                        .relTol     -- was hardcoded 0.0
            ctl_.bicgCheckEvery,
            eUseGS_,            //                        .solver smoothSolver -- was hardcoded false
            nullptr,
            nullptr,
            &kineticSrc,
            alphaEffBnd.size() ? &alphaEffBnd : nullptr);

        // OF's EEqn.H ends with thermo.correct(), so its PRESSURE equation sees psi/rho/mu updated from the
        // just-solved he. brae updated thermo only after the pressure step, leaving the pEqn on the previous
        // iteration's psi and rho. T/psi/mu/alpha depend on he alone (not p), so doing it here is exactly
        // OF's ordering; rho still gets refreshed with the NEW p and relaxed below, as OF does at the end
        // of pEqn.H/pcEqn.H.
        deviceThermoUpdate(th_, dp_, tc_);

        correctPressureVelocity(res);

        // pressureControl::limit(p), in OF's position: after the pressure solve and the U correction,
        // BEFORE rho is recomputed. Clamping after rho would let one NaN density through per iteration.
        if (rc_.limitMaxP || rc_.limitMinP)
        {
            // BRAE_DUMP_PLIMIT: report the PRE-clamp extrema, which is what OF's pressureControl prints
            // ("pressureControl: p max ..."). Comparable line-for-line against OF's log, and the only way
            // to compare a pressure field that both codes then saturate at the same limits.
            if (std::getenv("BRAE_DUMP_PLIMIT"))
            {
                const std::vector<scalar> ph = dp_.host();
                scalar lo = ph.empty() ? 0.0 : ph[0], hi = lo;
                for (scalar v : ph) { lo = std::min(lo, v); hi = std::max(hi, v); }
                std::fprintf(stderr, "pressureControl(brae): p max %g  p min %g\n", hi, lo);
            }
            deviceLimitPressure(dp_,
                                rc_.limitMinP ? rc_.pMinLimit : -1e300,
                                rc_.limitMaxP ? rc_.pMaxLimit : 1e300);
        }

        // thermo.correct() then rho.relax(): p and he have both moved, so every T-dependent property is
        // stale until this runs, and the relaxation stops the outer loop oscillating on the raw update.
        deviceThermoUpdate(th_, dp_, tc_);
        deviceRhoRelax(th_, tc_);

        correctTurbulence();

        // alphat = mut/Prt, for the NEXT iteration's energy equation. Must follow correctTurbulence(),
        // which is what makes nut valid. No-op while laminar (nut empty -> alphat stays zero), so the
        // energy equation is unaffected until phase 4.1 gives the SST a compressible nut.
        deviceAlphat(th_, dnut_, tc_);

        // Continuity. NOTE: with a mass flux this is a MASS residual, not a volumetric one -- the number
        // means something different from the incompressible report even though the code is identical.
        // Flagged in phi-audit.md as an INTERPRET site.
        {
            DeviceBuffer<scalar> divPhi;
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            DeviceBuffer<scalar> R;
            deviceHadamard(R, divPhi, dm.V);
            const scalar sumV = deviceDot(dm.V, ones_) + 1e-300;
            res.contLocal = deviceSumMag(R) / sumV;
            res.contGlobal = deviceDot(R, ones_) / sumV;
        }
        return res;
    }

    DeviceSimpleResidual DeviceSimpleSolver::pimpleStep(scalar deltaT, int nOuterCorrectors, int nCorrectors)
    {
        const DeviceMesh& dm = dm_;
        advanceTime(deltaT);                                  // store oldTime() + set deltaT/deltaT0 for ddtCoeffs()
        DeviceSimpleResidual res;
        const int nOuter = nOuterCorrectors > 0 ? nOuterCorrectors : 1;
        const int nCorr  = nCorrectors      > 0 ? nCorrectors      : 1;
        for (int oc = 0; oc < nOuter; ++oc)
        {
            solveMomentumPredictor(res);                     // momentum incl. implicit ddt
            for (int pc = 0; pc < nCorr; ++pc)               // PIMPLE pressure (inner) correctors
                correctPressureVelocity(res);
            correctTurbulence();
        }
        // continuity errors on the corrected flux (identical to step()).
        {
            DeviceBuffer<scalar> divPhi;
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            DeviceBuffer<scalar> R;
            deviceHadamard(R, divPhi, dm.V);
            const scalar sumV = deviceDot(dm.V, ones_) + 1e-300;
            res.contLocal  = deviceSumMag(R) / sumV;
            res.contGlobal = deviceDot(R, ones_) / sumV;
        }
        return res;
    }

    void DeviceSimpleSolver::setMRF(
        const MRFZone& z,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<std::string>& nonRotating)
    {
        mrf_ = buildDeviceMRF(z, m, g, fvp_, nonRotating);
        deviceMrfApplyFrameFlux(mrf_, +1.0, phiInt_, phiBnd_);     // initial phi -> relative (OF createFields)
        // NOTE: MRF is NOT applied to the cyclic/cyclicAMI interface flux. Ground truth: OF never combines MRF with
        // cyclicAMI (0/12 MRF tutorials use AMI, all use conformal multi-zone meshes; rotating AMI is done by a
        // MOVING MESH in transient solvers, not MRF). So there is no MRF x AMI pattern to support here.
    }

    void DeviceSimpleSolver::setFvOptions(const FvOptionsData& fo)
    {
        hasFvoMom_ = fo.hasMomentum;
        if (fo.hasMomentum)
        {
            for (int c = 0; c < 3; ++c)
                fvoMomSu_[c].copyFrom(fo.momSu[c]);
            if (!fo.momSp.empty()) fvoMomSp_.copyFrom(fo.momSp);
        }
        for (const auto& s : fo.scaSu)
            fvoScaSu_[s.first].copyFrom(s.second);   // scalar (k/epsilon/...) sources
        for (const auto& s : fo.scaSp)
            fvoScaSp_[s.first].copyFrom(s.second);
        if (fo.porActive)
        {
            por_.active = true;
            por_.d = fo.porD;
            por_.f = fo.porF;
            por_.cells.copyFrom(fo.porCells);
        }
        if (fo.mvfActive)
        {
            mvfActive_ = true;
            mvfRelax_ = fo.mvfRelax;
            const scalar m = std::sqrt(fo.mvfUbar.x*fo.mvfUbar.x + fo.mvfUbar.y*fo.mvfUbar.y + fo.mvfUbar.z*fo.mvfUbar.z);
            mvfUbarMag_ = m;
            mvfFlowDir_ = m > 0 ? vector{fo.mvfUbar.x/m, fo.mvfUbar.y/m, fo.mvfUbar.z/m} : vector{1,0,0};
            std::vector<scalar> m01(nC_, fo.mvfCells.empty() ? 1.0 : 0.0);   // selectionMode all -> ones; cellZone -> 1 in zone
            for (label c : fo.mvfCells)
                if (c >= 0 && c < nC_) m01[c] = 1.0;
            mvfMask01_.copyFrom(m01);
            deviceHadamard(mvfMaskV_, dm_.V, mvfMask01_);                     // maskV = V in the selection (0 else)
            mvfVtot_ = deviceSumMag(mvfMaskV_);                               // total selection volume
        }
        if (fo.limUActive)
        {
            limUActive_ = true;
            limUMax_ = fo.limUMax;
            std::vector<label> lc = fo.limUCells;
            if (lc.empty())   // selectionMode all
            {
                lc.resize(nC_);
                for (label c = 0; c < nC_; ++c)
                    lc[c] = c;
            }
            limUCells_.copyFrom(lc);
        }
        if (fo.vdcActive)
        {
            vdcActive_ = true;
            vdcUMax_ = fo.vdcUMax;
            vdcC_ = fo.vdcC;
            std::vector<label> vc = fo.vdcCells;
            if (vc.empty())   // selectionMode all
            {
                vc.resize(nC_);
                for (label c = 0; c < nC_; ++c)
                    vc[c] = c;
            }
            vdcCells_.copyFrom(vc);
        }
        if (fo.adActive)
        {
            adActive_ = true;
            adDiskDir_ = fo.adDiskDir;
            adArea_ = fo.adArea;
            adA_ = fo.adA;
            std::vector<scalar> dm01(nC_, 0.0);
            for (label c : fo.adDiskCells)
                if (c>=0 && c<nC_) dm01[c] = 1.0;
            DeviceBuffer<scalar> dmask;
            dmask.copyFrom(dm01);
            deviceHadamard(adMaskVDisk_, dm_.V, dmask);
            adVtot_ = deviceSumMag(adMaskVDisk_);   // V in disk / total disk volume
            std::vector<scalar> mon(nC_, 0.0);
            for (label c : fo.adMonitorCells)
                if (c>=0 && c<nC_) mon[c] = 1.0;
            adMonMask01_.copyFrom(mon);
            adNmon_ = (scalar)fo.adMonitorCells.size();
        }
        fvoCount_ = fo.count;
    }

}  // namespace brae
