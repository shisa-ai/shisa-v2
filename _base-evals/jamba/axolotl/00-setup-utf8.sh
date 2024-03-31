#!/bin/bash

# Check if the system has the necessary locales package installed
if ! dpkg -s locales >/dev/null 2>&1; then
	    echo "Installing locales package..."
	        apt-get update
		    apt-get install -y locales
fi

# Check if the en_US.UTF-8 locale is generated
if ! locale -a | grep -q "en_US.utf8"; then
	    echo "Generating en_US.UTF-8 locale..."
	        locale-gen en_US.UTF-8
fi

# Set the default locale to en_US.UTF-8 in the system-wide locale configuration file
echo 'LANG="en_US.UTF-8"' > /etc/default/locale
echo 'LC_ALL="en_US.UTF-8"' >> /etc/default/locale

# Source the system-wide locale configuration file
source /etc/default/locale

echo "UTF-8 setup completed."
