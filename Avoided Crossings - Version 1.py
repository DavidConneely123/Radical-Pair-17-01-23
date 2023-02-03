import numpy as np

from RadicalPair import *
from Programs import *



N5 = RadicalA('N5', 1, np.array([[-0.0989,0.0039,0.0],[0.0039,-0.0881,0.0],[0.0,0.0,1.7569]])*28.024951*1e6)
N10 = RadicalA('N10', 1, np.array([[-0.0190,-0.0048,0.0],[-0.0048,-0.0196,0.0],[0.0,0.0,0.6046]])*28.024951*1e6)
H8_1 = RadicalA('H8_1', 1/2, np.array([[0.4399,0.0,0.0],[0.0,0.4399,0.0],[0.0,0.0,0.4399]])*28.024951*1e6)
H8_2 = RadicalA('H8_2', 1/2, np.array([[0.4399,0.0,0.0],[0.0,0.4399,0.0],[0.0,0.0,0.4399]])*28.024951*1e6)

N1 = RadicalB('N1', 1, np.array([[-0.0336,0.0924,-0.1354],[0.0924,0.3303,-0.5318],[-0.1354,-0.5318,0.6680]])*28.024951*1e6)
H1 = RadicalB('H1', 1/2, np.array([[-0.9920,-0.2091,-0.2003],[-0.2091,-0.2631,0.2803],[-0.2003,0.2803,-0.5398]])*28.024951*1e6)
H2 = RadicalB('H2', 1/2, np.array([[-0.2843,0.1757,0.1525],[0.1757,-0.2798,0.0975],[0.1525,0.0975,-0.2699]])*28.024951*1e6)
H4 = RadicalB('H4', 1/2, np.array([[-0.5596,-0.1956,-0.1657],[-0.1956,-0.4020,0.0762],[-0.1657,0.0762,-0.5021]])*28.024951*1e6)

RadicalA.add_all_to_simulation()
RadicalB.add_all_to_simulation()


def Avoided_Crossings(centre_angle, angle_range_width, resolution):
    for theta in np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, resolution):
        Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, 0))
        eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)
        eigenvalues = eigenvalues /1e6
        plt.scatter(np.linspace(theta,theta,len(eigenvalues)), eigenvalues)


fig, ax = plt.subplots()

def Theta_Dependence_Of_Resonance_Effects(centre_angle, angle_range_width, resolution, save = False, file_name = None):
    resonance_effects =[]
    angle_range =  np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, resolution)
    for theta in angle_range:
        histogram_heights = Histogram_Heights_Single_Field_Direction(theta, 0, 120)[0]
        resonance_effect = np.sum(histogram_heights)
        resonance_effects.append(resonance_effect)
        print(f'Completed Theta = {theta}')
    plt.plot(angle_range,resonance_effects)

    if save:
        save_array = np.array([resonance_effects, angle_range])
        np.save(file_name, save_array)

def Resonance_Effect_new(theta,phi=0,rf_field_strength=1e-5,dipolar=False,rf_perpendicular=True):
    Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, 0,dipolar=dipolar))

    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)  # NB np.linalg.eigh gives [w[i]] eigenvalues in ascending order and a matrix with eigenvectors as COLUMNS not ROWS !!!
    eigenvalues = eigenvalues / 1e6  # Converting Eigenvalues to MHz

    Singlet_Projection_Matrix = eigenvectors.conj().T @ Singlet_Projection_total_basis() @ eigenvectors
    Populations = np.diag(Singlet_Projection_Matrix)

    if rf_perpendicular:
        Transition_Hamiltonian = eigenvectors.conj().T @ H_perp_total_basis(rf_field_strength, theta,phi) @ eigenvectors

    if not rf_perpendicular:
        Transition_Hamiltonian = eigenvectors.conj().T @ H_zee_total_basis(rf_field_strength, theta, phi) @ eigenvectors

    resonance_effect = 0
    for i in range(len(eigenvalues)):
        for j in range(len(eigenvalues)):
            resonance_effect += np.square(Populations[i] - Populations[j]) * np.square(np.abs(Transition_Hamiltonian[i, j]))

    return resonance_effect


def Resonance_Effect_range(centre_angle,angle_range_width,resolution,phi=0,rf_field_strength=1e-5,dipolar=False,rf_perpendicular=True, save=False, file_name=None):
    angle_range = np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, resolution)

    pool = multiprocessing.Pool(4)
    resonance_effects = pool.map(functools.partial(Resonance_Effect_new, phi=phi,rf_field_strength=rf_field_strength,dipolar=dipolar,rf_perpendicular=rf_perpendicular), angle_range)

    if save:
        save_array = np.array([resonance_effects, angle_range])
        np.save(file_name, save_array)

    plt.plot(angle_range,resonance_effects)



def Resonance_Effect(centre_angle,angle_range_width,resolution,phi=0,rf_field_strength=1e-5,dipolar=False,rf_perpendicular=True, save=False, file_name=None):
    resonance_effects = []
    angle_range = np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, resolution)

    for theta in angle_range:
        Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, 0,dipolar=dipolar))

        eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)  # NB np.linalg.eigh gives [w[i]] eigenvalues in ascending order and a matrix with eigenvectors as COLUMNS not ROWS !!!
        eigenvalues = eigenvalues / 1e6  # Converting Eigenvalues to MHz

        Singlet_Projection_Matrix = eigenvectors.conj().T @ Singlet_Projection_total_basis() @ eigenvectors
        Populations = np.diag(Singlet_Projection_Matrix)

        if rf_perpendicular:
            Transition_Hamiltonian = eigenvectors.conj().T @ H_perp_total_basis(rf_field_strength, theta,phi) @ eigenvectors

        if not rf_perpendicular:
            Transition_Hamiltonian = eigenvectors.conj().T @ H_zee_total_basis(rf_field_strength, theta, phi) @ eigenvectors

        resonance_effect = 0
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                    resonance_effect += np.square(Populations[i] - Populations[j]) * np.square(np.abs(Transition_Hamiltonian[i, j]))
        resonance_effects.append(resonance_effect)
        print(f'Completed Theta = {theta}')

        if save:
            save_array = np.array([resonance_effects, angle_range])
            np.save(file_name, save_array)

    plt.plot(angle_range,resonance_effects)

'''
centre_angle = 0.240
angle_range_width = 0.5
angle_range_2 = np.linspace(centre_angle - angle_range_width, centre_angle + angle_range_width, 100)
for theta in angle_range_2:
    Histogram_Heights_Single_Field_Direction(theta, 0, 120, display_varying=True)
'''

#Resonance_Effect(np.pi/2, np.pi/2,100)
Resonance_Effect_range(np.pi/2,np.pi/2,100,save=True,file_name='RversusTheta_C10.npy')
plt.show()

