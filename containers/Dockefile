FROM obalasz/prokbert

# Switch to root to install git (if not already present) and then switch back
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user (same as in the prokbert image)
USER prokbert
WORKDIR /home/prokbert

# Clone the PhaStyle repository
RUN git clone https://github.com/nbrg-ppcu/PhaStyle.git


# Add the PhaStyle 'bin' folder to PATH so you can run `PhaStyle.py` directly
ENV PATH="/home/prokbert/PhaStyle/bin:${PATH}"


# Default to bash
CMD ["/bin/bash"]
