from triped import triped_leg, gimbal_joint, extend_motor
from trip_kinematics.Robot import inverse_kinematics
import csv

if __name__ == '__main__':

    filename_input = 'tests/experiments/inverse_3/input_data/matlab_foot_coordinates.csv'

    filename_output = 'tests/experiments/inverse_3/output_data/gimbal_extend.csv'

    input_x = []
    input_y = []
    input_z = []

    with open(filename_input, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            input_x.append(float(row[1]))
            input_y.append(float(row[2]))
            input_z.append(float(row[3]))

    output_rows = []
    tip = {'t1': 0, 't2': 0, 'ry': 0}

    for i in range(len(input_x)):
        gimbal_joint.pass_arguments_g([tip])
        extend_motor.set_actuated_state([{'LL_revolute_joint_ry': tip['ry']}])

        row = inverse_kinematics(
            triped_leg, [input_x[i], input_y[i], input_z[i]])
        output_rows.append([row[0][0][0]['t1'], row[0][0]
                            [0]['t2'], row[1][0][0]['LL_revolute_joint_ry']])
        tip['t1'] = row[0][0][0]['t1']
        tip['t2'] = row[0][0][0]['t2']
        tip['ry'] = row[1][0][0]['LL_revolute_joint_ry']

    with open(filename_output, 'w') as f:
        writer = csv.writer(f)
        for row in output_rows:
            writer.writerow(row)
