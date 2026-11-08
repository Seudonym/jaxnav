FROM osrf/ros:humble-desktop-full

# Install dependencies
RUN apt-get update && apt-get install -y ros-humble-turtlebot3-* ros-humble-navigation2 ros-humble-nav2-bringup && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update && apt-get install -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "jax[cuda13]"

# Set environment variables
ENV TURTLEBOT3_MODEL=burger
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
