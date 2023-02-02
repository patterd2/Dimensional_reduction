import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy.spatial import distance

#setting the seed to a fixed value to make sure runs of the code are "predictable"
#np.random.seed(0)

#! Global Parameters
N = 20 # number of agent populations (predators) is N+1, zooplankton
M = 80 # number of subject populations (prey) is M+1, phytoplankton
T = 400 # number of timesteps
dt = 0.05 # length of timestep for forward Euler

#! Mutation rate (i.e., how ofter you update a diet preference)
#P = 0.1 # mutation rate
Q = 0.05 # mutation strength
P = 2 # Deactivating mutation

#Kmax_array=(0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.3,0.5,1.0,3.0,5.0,7.5,10.,15.)
Kmax_array=(0.01,0.03,0.06,0.1,0.5,1.0,2.5,5.0,10.0)
nK=np.count_nonzero(Kmax_array)

nreps=8
ik=0
#initialization of the arrays that are averaged over replicates for a given Kmax:
meanBCav_Kmax=np.zeros((nK))
eigenBCav_Kmax=np.zeros((nK))
detBCav_Kmax=np.zeros((nK))
nsurvA_Kmax=np.zeros((nK))
nsurvS_Kmax=np.zeros((nK))
for Kmax in Kmax_array:
    print("Kmax=",Kmax)

    for rep in np.arange(0,nreps):
        print("Replicate=",rep+1)

        #initialization of the system:
        #! state variables (to save over time)
        A = np.zeros((N,T))
        S = np.zeros((M,T))
        A[:,0] = 0.2*np.random.rand(N) # RANDOM initial conditions
        S[:,0] = 0.1*np.random.rand(M) # RANDOM initial conditions

        #Initialization of sizes for the different types (right now, made up values for max and min sizes)
        smaxS=0.5#max size for phyplankton, in um
        sminS=0.01#min size for phytoplankton, in um
        smaxA=5#max size for zooplankton, in um
        sminA=0.5#min size for zooplankton, in um
        sizeA=(smaxA-sminA)*np.random.rand(N)+sminA
        sizeS=(smaxS-sminS)*np.random.rand(M)+sminS
        
        #Store the indices that order the sizeA array from smaller to larger:
        orderA = sorted(range(len(sizeA)), key=lambda k: sizeA[k])
        
        #Translating phytoplankton sizes into size bins (10 for now):
        nbins=10
        #10=asize*smaxS+bsize
        #0=asize*sminS+bsize
        asize=(nbins-0.)/(smaxS-sminS)
        bsize=0.-asize*sminS    

        #Assuming a linear relationship linking size and K, so that 0.05*Kmax is reached for sminS and Kmax for smaxS:
        #1*Kmax=smaxS*a+b
        #0.05*Kmax=sminS*a+b
        a=(1-0.05)*Kmax/(smaxS-sminS)
        b=1*Kmax-smaxS*a
        K = a*sizeS+b
        r = np.exp(-K) * 2

        # encounter and handling (should depend on size as well, so we use this naive approach for now):
        Alphamax = 3.05
        Alpha = np.ones((N,M)) # encounter rate
        #1*Alphamax=smaxA*a+b
        #0.05*Alphamax=sminA*a+b
        a=(1-0.05)*Alphamax/(smaxA-sminA)
        b=1*Alphamax-smaxA*a
        for i in np.arange(0,N):
            for j in np.arange(0,M):        
                Alpha[i,j] = a*sizeA[i]+b
        
        Hmax = 2.68
        H = np.ones((N,M)) # handling time
        #1*Hmax=sminA*a+b
        #0.05*Hmax=smaxA*a+b
        a=(1-0.05)*Hmax/(sminA-smaxA)
        b=1*Hmax-sminA*a
        for i in np.arange(0,N):
            for j in np.arange(0,M):        
                H[i,j] = a*sizeA[i]+b

        #Yield parameter (should be size-specific):
        #Assuming a linear relationship linking size and Y, so that 0.05*Ymax is reached for sminA and Ymax for smaxA:
        Ymax=1.0
        #1*Ymax=smaxA*a+b
        #0.05*Ymax=sminA*a+b
        a=(1-0.05)*Ymax/(smaxA-sminA)      
        b=1*Ymax-smaxA*a
        Y = np.ones((N,M))
#        Y = np.ones((N,M)) * 1.0
        for i in np.arange(0,N):
            for j in np.arange(0,M):
                Y[i,j]=a*sizeA[i]+b
        
        #Mortality for zooplankton:
        m = np.ones(N) * 0.1 # natural mortality of agent populations

        # diet preference matrix (I see this as the proportion of feeding time dedicated to searching for a particular prey
        Phi = np.zeros((N,M,T)) 

        # random diet pref
        #Phi[:,:,0] = np.random.rand(N,M)

        # uniform diet pref
        Phi[:,:,0] = 1

        # normalize to row stochastic
        Phi[:,:,0] = Phi[:,:,0]/Phi[:,:,0].sum(axis=1)[:,None] # important to row normalize to sum to 1

        #! dynamic variables (not to save over time)
        dA = np.zeros(N)
        dS = np.zeros(M)
        DIET = np.zeros((N,M,T-1))
        PREF = np.zeros((N,M,T-1))
        BC = np.zeros((N,N,T-1))
        sm = np.zeros(N)
        handling = np.zeros(N)
        contact = np.zeros((N,M))
        feeding = np.zeros((N,M))
        growth = np.zeros((N,M))
        A_grow = np.zeros(N)
        S_grow = np.zeros(N)
        A_die = np.zeros(M)
        S_die = np.zeros(N)

        DIET_av = np.zeros((N,M))
        BC_av = np.zeros((N,N))

        #! Dynamics
        for t in np.arange(0,T-1):

            # loop through agent and subject populations to get feeding matrix
            for i in np.arange(0,N):

                # handling
                handling = 1 + np.sum(Phi[i,:,t]*Alpha[i,:]*H[i,:]*S[:,t])

                for j in np.arange(0,M):
                    # contact
                    contact[i,j] = Phi[i,j,t]*Alpha[i,j]*S[j,t]*A[i,t]
                
                    # feeding
                    feeding[i,j] = contact[i,j] / handling
                    
                    #growth
                    growth[i,j] = Y[i,j]*feeding[i,j]

            # Growth of agent population
            A_grow = np.sum(growth,1)

            # Realized diet matrix
            #DIET[:,:,t] = growth / A_grow[:, np.newaxis]
            #Accounting for the possibility of zoo or phytoplankton extintion:
            for i in np.arange(0,N):
                if A_grow[i]>0:
                    for j in np.arange(0,M):
                        DIET[i,j,t] = growth[i,j] / A_grow[i]
            
            #Diet preference (the part below should normalize the realized diet for each predator, i.e. take the sum of all the DIET elements for a given predator i, and use it to divide each of its DIET elements by it so that we get the proportion of predator i's diet that prey j represents)
        #    for i in np.arange(0,N):
        #        sm = np.sum(DIET[i,:,t])
        #        PREF[i,:,t]=DIET[i,:,t]/sm
            #Calculate the Bray-Curtis dissimilarity (or distance) between each pair of zooplankton types:
        #    for i in np.arange(0,N):
        #        for j in np.arange(0,N):
        #            BC[i,j,t]=distance.braycurtis(DIET[i,:,t],DIET[j,:,t])
            # agent mortality
            A_die = m * A[:,t]

            # agent population dynamics
            dA = (A_grow - A_die) * dt

            # Death of subject population
            S_die = np.sum(feeding,0)
                    
            # growth
            S_grow = r * (1 - (S[:,t]/K)) * S[:,t]

            # dynamics
            dS = (S_grow - S_die) * dt
                
            ## Update
            a = A[:,t] + dA
            s = S[:,t] + dS

            ## Preventing the atto-fox:
            a[np.where(a<10**-8)] = 0
            s[np.where(s<10**-8)] = 0

            ## Update
            A[:,t+1] = a
            S[:,t+1] = s

            ## Update diet preference
            Phi[:,:,t+1] = Phi[:,:,t]
            #if np.random.random() < P:
            #     # find a random agent (predator)
            #     i = np.random.randint(0,N,1)

            #     # find prey it ate the least
            #     j = np.argmin(DIET[i,:,t])
            #     k = np.argmax(DIET[i,:,t])

            #     # reduce diet pref for this prey and give to others
            #     ddiet = np.random.random()*Q*Phi[i,j,t]

                # update diet preference
            #     Phi[i,j,t+1] = Phi[i,j,t] - ddiet
            #     Phi[i,k,t+1] = Phi[i,k,t] + ddiet
        
        #Calculate the average diet preference over the last 10% of total duration of the simulation:
        for t in np.arange(int(0.9*(T-1)),T-1):
            DIET_av[:,:]=DIET_av[:,:]+DIET[:,:,t]

        DIET_av[:,:]=DIET_av[:,:]/(T-1-(0.9*(T-1)))

        #Mapping DIET_av into a size-based portfolio for each zooplankton. We also order zooplankton from smallest to largest:
        port=np.zeros((N,nbins))
        points=np.zeros((N,nbins))
        Neff=-1
        for ii in np.arange(0,N):
            i=orderA[ii]
            if A[i,T-1]>0:
                Neff=Neff+1
                for j in np.arange(0,M):
                    k=int(asize*sizeS[j]+bsize)
                    port[Neff,k]=port[Neff,k]+DIET_av[i,j]
                    points[Neff,k]=points[Neff,k]+1

        for i in np.arange(0,Neff):
            for j in np.arange(0,nbins):
                if points[i,j] > 0: port[i,j]=port[i,j]/points[i,j]

        #Calculate the Bray-Curtis dissimilarity matrix of that average diet preference matrix:
        for i in np.arange(0,Neff):
            for j in np.arange(0,Neff):
                BC_av[i,j]=distance.braycurtis(port[i,:]+0.0000001,port[j,:]+0.0000001)

        #Storing BC stats for the averages over replicates for this Kmax:
        dummy,dummyv = np.linalg.eig(BC_av[:,:])
        eigenBC=max(dummy)
        meanBC=np.sum(BC_av[:,:])/(N*N)
        detBC=np.linalg.det(BC_av[:,:])

        #Count how many zooplankton survived

        survA = A[:,T-1] > 0
        survS = S[:,T-1] > 0
        nsurvA_Kmax[ik] = nsurvA_Kmax[ik] + survA.sum()/(N+0.)
        nsurvS_Kmax[ik] = nsurvS_Kmax[ik] + survS.sum()/(M+0.)

        #print(rep,meanBC,eigenBC,detBC,survA.sum()/(N+0.),Neff)

        meanBCav_Kmax[ik]=meanBCav_Kmax[ik]+meanBC
        eigenBCav_Kmax[ik]=eigenBCav_Kmax[ik]+eigenBC
        detBCav_Kmax[ik]=detBCav_Kmax[ik]+detBC
        
        #### Plot
#        plt.figure(figsize=[8,5])
#        colors_A = plt.cm.Reds(np.linspace(0,1,M+1))
#        for i in np.arange(0,N):
#            plt.plot(A[i,:],label="Subject",color=colors_A[i])
#            plt.xlabel("Time")
#            plt.ylabel("Zooplankton Abundance")
#            plt.savefig("./Figs/zooplankton_Kmax_"+str(Kmax)+"_rep_"+str(rep)+".png",dpi=300,bbox_inches='tight')
#            
#        #plt.show()
#        plt.close()
#
#        plt.figure(figsize=[8,5])
#        colors_S = plt.cm.Blues(np.linspace(0,1,M+1))
#        colors_S = colors_S[1:,:]
#        for i in np.arange(0,M):
#            plt.plot(S[i,:],label="Subject",color=colors_S[i])
#            plt.xlabel("Time")
#            plt.ylabel("Phytoplankton Abundance")
#            plt.savefig("./Figs/phytoplankton_Kmax_"+str(Kmax)+"_rep_"+str(rep)+".png",dpi=300,bbox_inches='tight')
#        #plt.show()
#        plt.close()
#        
#        ## Plot heatmap of diets:
#        plt.imshow(DIET_av[:,:])
#        plt.clim(vmin=0,vmax=1.0)
#        plt.colorbar()
#        plt.ylabel("Zooplankton type")
#        plt.xlabel("Phytoplankton size class")
#        plt.title("Realized diet")
#        plt.savefig("./Figs/heatmap_diet_Kmax_"+str(Kmax)+"_rep_"+str(rep)+".png",dpi=300,bbox_inches='tight')
##        plt.show()
#        plt.close()
#
#        ## Plot heatmap of portfolios (with zooplankton ordered from smallest to largest):
#        plt.imshow(port[orderA,:])
#        plt.clim(vmin=0,vmax=1.0)
#        plt.colorbar()
#        plt.ylabel("Zooplankton type")
#        plt.xlabel("Phytoplankton size class")
#        plt.title("Realized portfolio")
#        plt.savefig("./Figs/heatmap_portfolio_Kmax_"+str(Kmax)+"_rep_"+str(rep)+".png",dpi=300,bbox_inches='tight')
##        plt.show()
#        plt.close()
#        
#        ## Plot heatmap of (ordered) BC matrix:
#        plt.imshow(BC_av[:,:])
#        plt.clim(vmin=0,vmax=1.0)
#        plt.colorbar()
#        plt.ylabel("Zooplankton type")
#        plt.xlabel("Zooplankton type")
#        plt.title("Bray-Curtis dissimilarity")
#        plt.savefig("./Figs/heatmap_BC_Kmax_"+str(Kmax)+"_rep_"+str(rep)+".png",dpi=300,bbox_inches='tight')
##        plt.show()
#        plt.close()        


#    print(ik,Kmax,meanBCav_Kmax[ik]/nreps,eigenBCav_Kmax[ik]/nreps,detBCav_Kmax[ik]/nreps,nsurvA_Kmax[ik]/nreps)
    ik=ik+1

#np.savetxt('data_output', np.c_[Kmax_array,nsurvA_Kmax/nreps,nsurvS_Kmax/nreps,meanBCav_Kmax/nreps,eigenBCav_Kmax/nreps,detBCav_Kmax/nreps],fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

plt.plot(Kmax_array,meanBCav_Kmax/nreps, marker='o')
plt.ylabel("mean BC dissimilarity")
plt.xlabel("Max carrying capacity")
plt.savefig("./Figs/Fig_meanBC_vs_Kmax.png",dpi=300,bbox_inches='tight')
plt.close()

plt.plot(Kmax_array,eigenBCav_Kmax/nreps, marker='o')
plt.ylabel("Eigenvalue BC dissimilarity")
plt.xlabel("Max carrying capacity")
plt.savefig("./Figs/Fig_eigenBC_vs_Kmax.png",dpi=300,bbox_inches='tight')
plt.close()

plt.plot(Kmax_array,detBCav_Kmax/nreps, marker='o')
plt.ylabel("Determinant BC dissimilarity")
plt.xlabel("Max carrying capacity")
plt.savefig("./Figs/Fig_detBC_vs_Kmax.png",dpi=300,bbox_inches='tight')
plt.close()

plt.plot(Kmax_array,nsurvA_Kmax/nreps, marker='o')
plt.ylabel("Percentage of surviving zooplankton types")
plt.xlabel("Max carrying capacity")
plt.savefig("./Figs/Fig_Zsurv_vs_Kmax.png",dpi=300,bbox_inches='tight')
plt.close()

plt.plot(Kmax_array,nsurvS_Kmax/nreps, marker='o')
plt.ylabel("Percentage of surviving phytoplankton types")
plt.xlabel("Max carrying capacity")
plt.savefig("./Figs/Fig_Psurv_vs_Kmax.png",dpi=300,bbox_inches='tight')
plt.close()

#CLOSER LOOK TO SAME PLOTS:

#plt.plot(Kmax_array,meanBCav_Kmax/nreps, marker='o')
#plt.ylabel("mean BC dissimilarity")
#plt.xlabel("Max carrying capacity")
#plt.xlim([0.005,0.5])
#plt.savefig("./Figs/Fig_meanBC_vs_Kmax_ZOOM.png",dpi=300,bbox_inches='tight')
#plt.close()

#plt.plot(Kmax_array,eigenBCav_Kmax/nreps, marker='o')
#plt.ylabel("Eigenvalue BC dissimilarity")
#plt.xlabel("Max carrying capacity")
#plt.xlim([0.005,0.5])
#plt.savefig("./Figs/Fig_eigenBC_vs_Kmax_ZOOM.png",dpi=300,bbox_inches='tight')
#plt.close()

#plt.plot(Kmax_array,detBCav_Kmax/nreps, marker='o')
#plt.ylabel("Determinant BC dissimilarity")
#plt.xlabel("Max carrying capacity")
#plt.xlim([0.005,0.5])
#plt.savefig("./Figs/Fig_detBC_vs_Kmax_ZOOM.png",dpi=300,bbox_inches='tight')
#plt.close()

#plt.plot(Kmax_array,nsurvA_Kmax/nreps, marker='o')
#plt.ylabel("Percentage of surviving zooplankton types")
#plt.xlabel("Max carrying capacity")
#plt.xlim([0.005,0.5])
#plt.savefig("./Figs/Fig_Zsurv_vs_Kmax_ZOOM.png",dpi=300,bbox_inches='tight')
#plt.close()

plt.plot(Kmax_array,nsurvS_Kmax/nreps, marker='o')
plt.ylabel("Percentage of surviving phytoplankton types")
plt.xlabel("Max carrying capacity")
plt.xlim([0.0,5.0])
plt.savefig("./Figs/Fig_Psurv_vs_Kmax_ZOOM.png",dpi=300,bbox_inches='tight')
plt.close()
