class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    boundary_awareness = NumericProperty(0)  # New property for boundary awareness

    def __init__(self, **kwargs):
        super(Car, self).__init__(**kwargs)
        self.sensor_range = 20  # Range of sensors
        self.sensor_angles = [-30, 0, 30]  # Angles for sensors in degrees
        self.sensor_values = [0, 0, 0]  # Store raw sensor values
        self.last_position = None
        self.stuck_counter = 0
        self.max_stuck_steps = 10

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # Update sensors with improved positioning
        self.update_sensors()

        # Check if car is stuck
        if self.last_position:
            distance_moved = Vector(self.pos).distance(self.last_position)
            if distance_moved < 0.1:  # Car barely moved
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.last_position = Vector(self.pos)

        # Calculate boundary awareness
        self.calculate_boundary_awareness()

    def update_sensors(self):
        # Update sensor positions based on car's position and angle
        for i, angle in enumerate(self.sensor_angles):
            sensor_angle = self.angle + angle
            sensor_x = self.center_x + self.sensor_range * cos(radians(sensor_angle))
            sensor_y = self.center_y + self.sensor_range * sin(radians(sensor_angle))
            
            # Update sensor position
            if i == 0:
                self.sensor1 = (sensor_x, sensor_y)
            elif i == 1:
                self.sensor2 = (sensor_x, sensor_y)
            else:
                self.sensor3 = (sensor_x, sensor_y)

    def calculate_boundary_awareness(self):
        # Calculate how close the car is to boundaries
        margin = 20  # Margin from boundaries
        x_awareness = min(self.x / margin, (self.parent.width - self.x) / margin)
        y_awareness = min(self.y / margin, (self.parent.height - self.y) / margin)
        self.boundary_awareness = min(x_awareness, y_awareness)

    def get_sensor_values(self, sand):
        # Get sensor values with improved accuracy
        for i, sensor in enumerate([self.sensor1, self.sensor2, self.sensor3]):
            x = int(sensor[0])
            y = int(sensor[1])
            if 0 <= x < sand.shape[0] and 0 <= y < sand.shape[1]:
                self.sensor_values[i] = sand[x, y]
            else:
                self.sensor_values[i] = 1  # Treat out of bounds as obstacle

        # Normalize sensor values
        self.signal1 = self.sensor_values[0] / 255.0
        self.signal2 = self.sensor_values[1] / 255.0
        self.signal3 = self.sensor_values[2] / 255.0

    def is_stuck(self):
        return self.stuck_counter >= self.max_stuck_steps 