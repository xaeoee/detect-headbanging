import matplotlib.pyplot as plt

def plot_results(angle_changes, direction_changes, direction_change_counts, x_cord, y_cord, z_cord):
    plt.figure(figsize=(10, 4))
    plt.plot(x_cord, label='X Change')
    plt.xlabel('Frame')
    plt.legend()

    plt.figure(figsize=(10, 4))
    plt.plot(y_cord, label='Y Change')
    plt.xlabel('Frame')
    plt.legend()

    plt.figure(figsize=(10, 4))
    plt.plot(z_cord, label='Z Change')
    plt.xlabel('Frame')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(angle_changes, label='Angle Change')
    plt.xlabel('Frame')
    plt.ylabel('Angle Change (degrees)')
    plt.title('Angle Change Over Frames')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(direction_changes, label='Direction Change', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Direction Change')
    plt.title('Direction Change Over Frames')
    plt.yticks([-2, -1, 0, 1, 2], ['Right','Down', 'No_change', 'Up', 'Left'])
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(direction_change_counts, label='Direction Change Count', color='green')
    plt.xlabel('Frame')
    plt.ylabel('Direction Change Count')
    plt.title('Direction Change Count Over Frames')
    plt.legend()
    plt.show()
