import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
import numpy as np

from RadicalPair import *
from Programs import *

from scipy.spatial.transform import Rotation as R

r = R.from_euler('x', 0, degrees=True)
r = r.as_matrix()


#### Quantum Needle Tensors ######
#N5 = RadicalA('N5', 1, r@np.array([[-0.0989,0.0039,0.0],[0.0039,-0.0881,0.0],[0.0,0.0,1.7569]])*28.024951*1e6)
#N10 = RadicalA('N10', 1, r@np.array([[-0.0190,-0.0048,0.0],[-0.0048,-0.0196,0.0],[0.0,0.0,0.6046]])*28.024951*1e6)
#H8_1 = RadicalA('H8_1', 1/2, np.array([[0.4399,0.0,0.0],[0.0,0.4399,0.0],[0.0,0.0,0.4399]])*28.024951*1e6)

#N1 = RadicalB('N1', 1, r@np.array([[-0.0336,0.0924,-0.1354],[0.0924,0.3303,-0.5318],[-0.1354,-0.5318,0.6680]])*28.024951*1e6)
#H1 = RadicalB('H1', 1/2, r@np.array([[-0.9920,-0.2091,-0.2003],[-0.2091,-0.2631,0.2803],[-0.2003,0.2803,-0.5398]])*28.024951*1e6)
#H_b_1 = RadicalB('H_b_1', 1/2, np.array([[1.5808,-0.0453,-0.0506],[-0.0453,1.5575,0.0988],[-0.0506,0.0988,1.6752]])*28.024951*1e6)




### 2022 Paper (Alice) Tensors ####
#N5 = RadicalA('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
N5_mod = RadicalA('N5_mod', 1, np.array([[0, -0.08, 0], [-0.08, 0, 0], [0, 0, 49.24]]) * 1e6)
#N10 = RadicalA('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6)
N10_mod = RadicalA('N10', 1, np.array([[0, -0.06, 0.00], [-0.06, 0, 0.00], [0.00, 0.00, 16.94]]) * 1e6)
H8_1 = RadicalA('H8_1', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
#H6 = RadicalA('H6', 1/2, np.array([[-5.63, 0.92, 0.00], [0.92, -14.77, 0.00], [0.00, 0.00, -12.15]])*1e6)

NE1 = RadicalB('NE1', 1, np.array([[-1.48,1.64,-1.29],[1.64,15.82,-15.83],[-1.29,-15.83,12.70]])*1e6)
HE1 = RadicalB('HE1', 1/2, np.array([[-28.05, 5.78,5.42],[5.78,-12.38,8.61],[5.42,8.61,-9.88]])*1e6)
HB1 = RadicalB('HB1', 1/2, np.array([[44.07,0.44,1.33],[0.44,42.47,1.77],[1.33,1.77,48.36]])*1e6)


RadicalA.add_all_to_simulation()
RadicalB.add_all_to_simulation()


def Avoided_Crossings(centre_angle, angle_range_width, resolution):
    for theta in np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, resolution):
        Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, 0))
        eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)
        eigenvalues = eigenvalues /1e6
        plt.scatter(np.linspace(theta,theta,len(eigenvalues)), eigenvalues)


fig, ax = plt.subplots()


def Resonance_Effect_new(theta,phi=0,rf_field_strength=1e-5,dipolar=False,rf_perpendicular=True, rf_arbitrary=False, theta2=0, phi2=0):
    Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, phi,dipolar=dipolar))

    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)  # NB np.linalg.eigh gives [w[i]] eigenvalues in ascending order and a matrix with eigenvectors as COLUMNS not ROWS !!!
    eigenvalues = eigenvalues / 1e6  # Converting Eigenvalues to MHz

    Singlet_Projection_Matrix = eigenvectors.conj().T @ Singlet_Projection_total_basis() @ eigenvectors
    Populations = np.diag(Singlet_Projection_Matrix)

    if rf_perpendicular:
        print('RF field is perpendicular to the static field')
        Transition_Hamiltonian = eigenvectors.conj().T @ H_perp_total_basis(rf_field_strength, theta,phi) @ eigenvectors

    if not rf_perpendicular:
        if not rf_arbitrary:
            print('RF field is parallel to the static field')
            Transition_Hamiltonian = eigenvectors.conj().T @ H_zee_total_basis(rf_field_strength, theta, phi) @ eigenvectors

        if rf_arbitrary:
            print(f'RF field is at an orientation (Theta = {theta2}, Phi = {phi2}) to the static field')
            Transition_Hamiltonian = eigenvectors.conj().T @ H_zee_total_basis(rf_field_strength, theta+theta2,phi+phi2) @ eigenvectors

    resonance_effect = 0
    for i in range(len(eigenvalues)):
        for j in range(len(eigenvalues)):
            resonance_effect += np.square(Populations[i] - Populations[j]) * np.square(np.abs(Transition_Hamiltonian[i, j]))

    return resonance_effect


def Resonance_Effect_range(centre_angle,angle_range_width,resolution,phi=0,rf_field_strength=1e-5,dipolar=False,rf_perpendicular=True, rf_arbitrary=False, theta2=0,phi2=0, save=False, file_name=None):
    angle_range = np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, resolution)

    pool = multiprocessing.Pool(4)
    resonance_effects = pool.map(functools.partial(Resonance_Effect_new, phi=phi,rf_field_strength=rf_field_strength,dipolar=dipolar,rf_perpendicular=rf_perpendicular, rf_arbitrary=rf_arbitrary,theta2=theta2,phi2=phi2), angle_range)

    if save:
        save_array = np.array([resonance_effects, angle_range])
        np.save(file_name, save_array)

    rad_to_deg = 180/np.pi
    angle_range = np.linspace(rad_to_deg*(centre_angle - angle_range_width), rad_to_deg*(centre_angle + angle_range_width), resolution)

    min_angle = angle_range[resonance_effects.index(min(resonance_effects))]

    plt.plot(angle_range,resonance_effects)
    plt.plot(np.linspace(min_angle,min_angle,100), np.linspace(0,max(resonance_effects),100), color='red', linestyle = '--')
    plt.ylim(0,max(resonance_effects))
    plt.xlim(min(angle_range),max(angle_range))


def Plot_RversusTheta(file_name, color = None, title = None, label = None, integrate = False, linewidth=1, scale = False):
    resonance_effects, angle_range = np.load(file_name)[0], np.load(file_name)[1]
    angle_range = [x*180/np.pi for x in angle_range]

    if scale:
        resonance_effects = [resonance_effect/np.max(resonance_effects) for resonance_effect in resonance_effects]

    if integrate:
        from scipy.integrate import simps
        integration = "{:.2e}".format(int(np.real(scipy.integrate.simps(resonance_effects))))
        label = label +' (' + str(integration)+ ')'

    plt.plot(angle_range,resonance_effects, color = color, label = label, linewidth = linewidth)
    plt.legend()
    plt.ylim(0, 1.1*max(resonance_effects))
    plt.xlim(min(angle_range), max(angle_range))

def Surface(resolution, color='red', save = False, file_name = None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Create a mesh-grid of the desired resolution

    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    thetaGrid, phiGrid = np.meshgrid(theta, phi)


    # The radius at each (theta,phi) is the resonance effect at that field direction

    r_array = np.empty((resolution, resolution))



    for i in range(resolution):
        for j in range(resolution):
            print(theta[i], phi[j])
            r_array[i, j] = Resonance_Effect_new(theta[j], phi=phi[i])

    if save:
        np.save(file_name, r_array)



    X = r_array * np.sin(thetaGrid) * np.cos(phiGrid)
    Y = r_array * np.sin(thetaGrid) * np.sin(phiGrid)
    Z = r_array * np.cos(thetaGrid)

    ax.plot_surface(X, Y ,Z , color=color)
    ax.set_box_aspect((np.ptp(X),np.ptp(Y),np.ptp(Z)))


def Plot_r_array(file_name, color = 'red', invert = False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Create a mesh-grid of the desired resolution

    r_array = np.load(file_name)
    resolution = int(np.sqrt(np.size(r_array)))

    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    thetaGrid, phiGrid = np.meshgrid(theta, phi)

    r_array = np.load(file_name)

    if invert:
        max = np.max(r_array)
        for i in range(resolution):
            for j in range(resolution):
                r_array[i][j] = (max - r_array[i][j])/max

    X = r_array * np.sin(thetaGrid) * np.cos(phiGrid)
    Y = r_array * np.sin(thetaGrid) * np.sin(phiGrid)
    Z = r_array * np.cos(thetaGrid)

    ax.plot_surface(X, Y, Z, color=color)
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    #ax.set_box_aspect((1, 1, 1))

viridis = matplotlib.cm.get_cmap('hsv')
norm = mpl.colors.Normalize(vmin=0,vmax=90)

#Resonance_Effect_range(np.pi/2,np.pi/2,2000, dipolar=False, save=True, file_name='RversusTheta_K14_2000.npy')


#Plot_RversusTheta('RversusTheta_K3_2000.npy', scale = True, color = viridis(0), label = 'N5 / N1')
#Plot_RversusTheta('RversusTheta_K4_2000.npy', scale = True, color = viridis(0.1), label = 'N5 / HE1')
#Plot_RversusTheta('RversusTheta_K7_2000.npy', scale=True, color = viridis(0.2), label = 'H6,H8_1/N1')
#Plot_RversusTheta('RversusTheta_K8_2000.npy', scale = True, color = viridis(0.3), label = 'N10 / HE1')
#Plot_RversusTheta('RversusTheta_K9_2000.npy', scale=True, color = viridis(0.4), label = 'N10 / N1')
#Plot_RversusTheta('RversusTheta_K10_2000.npy', scale = True, color = viridis(0.6), label = 'N5_mod / N1')

#Plot_RversusTheta('RversusTheta_K2_2000.npy', scale = True, color = viridis(0.25), label = '2+2')
#Plot_RversusTheta('Quantum Needle Data/Quantum Needle Testing: 10-02-23/RversusTheta_H4_2000.npy', scale = True, color = viridis(0.5), label = '3+3')
#Plot_RversusTheta('Quantum Needle Data/Quantum Needle Testing: 14-02-23/RversusTheta_K1_2000.npy', scale = True, color = viridis(0.75), label ='4+4')


#Plot_RversusTheta('RversusTheta_K12_2000.npy', scale = True, color = viridis(0.9), label = 'N5 / NE1, HE1')
#Plot_RversusTheta('RversusTheta_K11_2000.npy', scale = True, color = viridis(0.7), label = 'N5_mod / NE1, HE1')

#Plot_RversusTheta('RversusTheta_K2_2000.npy', scale = True, color = viridis(0.9), label = 'N5, N10, / NE1, HE1')
#Plot_RversusTheta('RversusTheta_K13_2000.npy', scale = True, color = viridis(0.7), label = ' N5_mod, N10_mod / NE1, HE1')

Plot_RversusTheta('Quantum Needle Data/Quantum Needle Testing: 10-02-23/RversusTheta_H4_2000.npy', scale=True, color = viridis(0.9), label = 'N5, N10, H8 / NE1, HE1, HB1')
Plot_RversusTheta('RversusTheta_K14_2000.npy', scale = True, color= viridis(0.7), label = 'N5_mod, N10_mod, H8 / NE1, HE1, HB1')

plt.show()
