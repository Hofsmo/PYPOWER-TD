from pypiw import systems


def main():
    '''
    Example demonstrating how to simulate the Modelica file
    '''

    #Creating the system object

    fmu_sys = systems.ModelicaSystem('C:/temp/model2.fmu','model2')
    # y = fmu_sys.time_response(True, 1, 10)

if __name__ == "__main__":
    main()
