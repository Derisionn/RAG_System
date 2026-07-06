#!/usr/bin/env bash
# Force upgrade pip
python -m pip install --upgrade pip

# Install dependencies, bypassing cache for the critical authentication packages
pip install --no-cache-dir python-jose>=3.3.0 bcrypt>=4.0.1 google-auth>=2.23.0 sendgrid>=6.10.0 email-validator>=2.1.0

# Install the rest of the requirements
pip install -r requirements.txt
