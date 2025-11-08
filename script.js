const express = require('express');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const twilio = require('twilio');
const axios = require('axios');
const app = express();
const port = 3000;

// Twilio Configuration - UPDATED WITH CORRECT CREDENTIALS
const twilioClient = twilio(
    'ACf60f450f29fabf5d4dd01680f2052f48',  // Your Account SID
    '614f4f07bfff358...'                   // ✅ NEW Auth Token from your screenshot
);
const twilioPhoneNumber = '+14787395985';  // Your Twilio phone number

// EmailJS Configuration
const EMAILJS_CONFIG = {
    serviceId: 'service_akash',
    templateId: 'template_akash',  
    publicKey: 'CaMVUkQYox6o96Q29'
};

// Middleware
app.use(express.json());
app.use(express.static('.'));

// Add CORS headers
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

// Enhanced Alert Message Generator (Matches your screenshot format)
function generateAlertMessage(transaction, type = 'sms') {
    const riskLevel = transaction.riskScore >= 60 ? 'HIGH RISK' : 
                     transaction.riskScore >= 30 ? 'MEDIUM RISK' : 'LOW RISK';
    
    const timestamp = new Date(transaction.timestamp * 1000).toLocaleString();
    
    if (type === 'sms') {
        return `🚨 FRAUD ALERT:\n\nTransaction $${transaction.amount.toLocaleString()} at ${transaction.location}.\nRisk: ${transaction.riskScore}% (${riskLevel}).\n\nAccount: ${transaction.accNo}.\nTimestamp: ${timestamp}.\n\nPlease verify immediately.\n\n---\nSent from Fraud Detection System`;
    } else {
        return `🚨 FRAUD ALERT - ${riskLevel}\n\nTRANSACTION DETAILS:\n• Amount: $${transaction.amount.toLocaleString()}\n• Location: ${transaction.location}\n• Account: ${transaction.accNo}\n• Risk Score: ${transaction.riskScore}%\n• Timestamp: ${timestamp}\n\nALERTS:\n${transaction.alerts.map(alert => `• ${alert}`).join('\n')}\n\nACTION REQUIRED: Please verify immediately.\n\n---\nFraud Detection System`;
    }
}

// API endpoint to call C backend
app.post('/api/process-transaction', async (req, res) => {
    const { accNo, amount, location, mobileNumber, emailAddress } = req.body;
    
    console.log(`📊 Processing transaction: Account ${accNo}, Amount $${amount}, Location ${location}`);
    
    // Execute the C program
    const command = `./fraudbackend ${accNo} ${amount} "${location}" "${mobileNumber}" "${emailAddress}"`;
    
    exec(command, async (error, stdout, stderr) => {
        if (error) {
            console.error('❌ C backend error:', error);
            console.log('🔄 Falling back to JavaScript simulation...');
            
            const jsResult = simulateBackend(accNo, amount, location, mobileNumber, emailAddress);
            
            // Send alerts based on risk
            await sendRiskBasedAlerts(jsResult.transaction);
            
            return res.json(jsResult);
        }
        
        try {
            const result = JSON.parse(stdout);
            console.log('✅ C backend result - Risk Score:', result.transaction.riskScore);
            
            // Send alerts based on risk
            await sendRiskBasedAlerts(result.transaction);
            
            res.json(result);
        } catch (parseError) {
            console.error('❌ Parse error:', parseError);
            res.status(500).json({ 
                success: false, 
                error: 'Invalid response from backend' 
            });
        }
    });
});

// Smart Alert System - SMS for medium risk, both for high risk
async function sendRiskBasedAlerts(transaction) {
    const riskScore = transaction.riskScore;
    
    console.log(`🚨 Risk-based alerts - Score: ${riskScore}%`);
    
    const results = {
        sms: { success: false, reason: 'Not triggered' },
        email: { success: false, reason: 'Not triggered' }
    };
    
    // Send SMS for medium risk and above
    if (riskScore >= 30) {
        console.log('📱 Sending SMS alert (risk ≥ 30%)...');
        results.sms = await sendSMSAlert(transaction);
    }
    
    // Send email for high risk
    if (riskScore >= 60) {
        console.log('📧 Sending email alert (risk ≥ 60%)...');
        results.email = await sendEmailAlert(transaction);
    }
    
    console.log('📋 Alert Results:', results);
    return results;
}

// Enhanced SMS Alert Function
async function sendSMSAlert(transaction) {
    try {
        const message = generateAlertMessage(transaction, 'sms');
        
        console.log('📱 Attempting to send Twilio SMS to:', transaction.phone);
        
        // Validate phone number format
        let phoneNumber = transaction.phone;
        if (!phoneNumber.startsWith('+')) {
            phoneNumber = '+' + phoneNumber.replace(/\D/g, '');
        }

        const twilioResponse = await twilioClient.messages.create({
            body: message,
            from: twilioPhoneNumber,
            to: phoneNumber
        });

        console.log('✅ SMS sent via Twilio. SID:', twilioResponse.sid);
        return { 
            success: true, 
            sid: twilioResponse.sid, 
            service: 'twilio',
            message: 'SMS alert sent successfully'
        };
    } catch (error) {
        console.error('❌ Twilio SMS error:', error.message);
        return { 
            success: false, 
            error: error.message,
            service: 'twilio'
        };
    }
}

// Enhanced Email Alert Function
async function sendEmailAlert(transaction) {
    try {
        const riskLevel = transaction.riskScore >= 60 ? 'HIGH RISK' : 'MEDIUM RISK';
        
        const emailParams = {
            to_email: transaction.email,
            to_name: 'Security Team',
            from_name: 'Fraud Detection System',
            subject: `🚨 Fraud Alert - ${riskLevel} - Account ${transaction.accNo}`,
            message: generateAlertMessage(transaction, 'email'),
            transaction_amount: `$${transaction.amount.toLocaleString()}`,
            transaction_location: transaction.location,
            risk_score: `${transaction.riskScore}%`,
            risk_level: riskLevel,
            account_number: transaction.accNo.toString(),
            timestamp: new Date(transaction.timestamp * 1000).toLocaleString()
        };

        console.log('📧 Attempting to send EmailJS email to:', transaction.email);
        
        const emailjsResponse = await axios.post(
            `https://api.emailjs.com/api/v1.0/email/send`,
            {
                service_id: EMAILJS_CONFIG.serviceId,
                template_id: EMAILJS_CONFIG.templateId,
                user_id: EMAILJS_CONFIG.publicKey,
                template_params: emailParams
            },
            {
                headers: {
                    'Content-Type': 'application/json'
                },
                timeout: 10000
            }
        );

        console.log('✅ Email sent via EmailJS. Status:', emailjsResponse.status);
        return { 
            success: true, 
            messageId: emailjsResponse.data, 
            service: 'emailjs',
            message: 'Email alert sent successfully'
        };
    } catch (error) {
        console.error('❌ EmailJS error:', error.response?.data || error.message);
        return { 
            success: false, 
            error: error.response?.data || error.message,
            service: 'emailjs'
        };
    }
}

// Manual alert endpoints
app.post('/api/send-sms', async (req, res) => {
    try {
        const { phoneNumber, message } = req.body;
        
        if (!phoneNumber || !message) {
            return res.status(400).json({ success: false, error: 'Phone number and message are required' });
        }

        console.log('📱 Manual SMS request to:', phoneNumber);
        
        let formattedPhone = phoneNumber;
        if (!formattedPhone.startsWith('+')) {
            formattedPhone = '+' + formattedPhone.replace(/\D/g, '');
        }

        const twilioResponse = await twilioClient.messages.create({
            body: message,
            from: twilioPhoneNumber,
            to: formattedPhone
        });

        res.json({ 
            success: true, 
            sid: twilioResponse.sid, 
            message: 'SMS sent successfully via Twilio',
            service: 'twilio'
        });
    } catch (error) {
        console.error('❌ SMS error:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message,
            service: 'twilio'
        });
    }
});

app.post('/api/send-email', async (req, res) => {
    try {
        const { email, subject, message } = req.body;
        
        if (!email || !subject || !message) {
            return res.status(400).json({ success: false, error: 'Email, subject, and message are required' });
        }

        console.log('📧 Manual Email request to:', email);
        
        const emailParams = {
            to_email: email,
            to_name: 'Recipient',
            from_name: 'Fraud Detection System',
            subject: subject,
            message: message
        };

        const emailjsResponse = await axios.post(
            `https://api.emailjs.com/api/v1.0/email/send`,
            {
                service_id: EMAILJS_CONFIG.serviceId,
                template_id: EMAILJS_CONFIG.templateId,
                user_id: EMAILJS_CONFIG.publicKey,
                template_params: emailParams
            }
        );

        res.json({ 
            success: true, 
            messageId: emailjsResponse.data,
            message: 'Email sent successfully via EmailJS',
            service: 'emailjs'
        });
    } catch (error) {
        console.error('❌ Email error:', error.response?.data || error.message);
        res.status(500).json({ 
            success: false, 
            error: error.response?.data || error.message,
            service: 'emailjs'
        });
    }
});

// Test endpoints
app.get('/api/test-backend', (req, res) => {
    res.json({ success: true, service: 'backend', status: 'active' });
});

app.get('/api/test-sms', async (req, res) => {
    try {
        const account = await twilioClient.api.accounts(twilioClient.accountSid).fetch();
        res.json({ 
            success: true, 
            service: 'sms', 
            status: 'active',
            provider: 'twilio',
            account: account.friendlyName
        });
    } catch (error) {
        console.error('❌ Twilio test failed:', error.message);
        res.json({ 
            success: false, 
            service: 'sms', 
            status: 'inactive',
            error: error.message,
            provider: 'twilio'
        });
    }
});

app.get('/api/test-email', async (req, res) => {
    try {
        const testResponse = await axios.get(`https://api.emailjs.com/api/v1.0/domain/check?user_id=${EMAILJS_CONFIG.publicKey}`);
        res.json({ 
            success: true, 
            service: 'email', 
            status: 'active',
            provider: 'emailjs'
        });
    } catch (error) {
        console.error('❌ EmailJS test failed:', error.message);
        res.json({ 
            success: false, 
            service: 'email', 
            status: 'inactive',
            error: error.message,
            provider: 'emailjs'
        });
    }
});

// Alert testing endpoint
app.post('/api/test-alert', async (req, res) => {
    try {
        const { phoneNumber, email, amount = 100000, riskScore = 30 } = req.body;
        
        const testTransaction = {
            phone: phoneNumber,
            email: email,
            amount: amount,
            location: 'Tokyo',
            riskScore: riskScore,
            accNo: 104,
            timestamp: Math.floor(Date.now() / 1000),
            alerts: ['High-value transaction', 'Location change detected']
        };

        console.log('🧪 Testing alert system...');
        
        const results = await sendRiskBasedAlerts(testTransaction);
        
        res.json({
            success: true,
            message: 'Alert test completed',
            results: results
        });
        
    } catch (error) {
        console.error('❌ Alert test error:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

// Service configuration endpoint
app.get('/api/services/config', (req, res) => {
    res.json({
        twilio: {
            accountSid: twilioClient.accountSid ? 'configured' : 'missing',
            phoneNumber: twilioPhoneNumber,
            status: 'configured'
        },
        emailjs: {
            serviceId: EMAILJS_CONFIG.serviceId,
            templateId: EMAILJS_CONFIG.templateId,
            publicKey: EMAILJS_CONFIG.publicKey,
            status: 'configured'
        },
        alertSystem: {
            smsThreshold: 30,
            emailThreshold: 60,
            status: 'active'
        }
    });
});

// JavaScript simulation fallback
function simulateBackend(accNo, amount, location, mobileNumber, emailAddress) {
    const alerts = [];
    let riskScore = 0;
    
    if (amount > 100000) {
        alerts.push("Very high-value transaction");
        riskScore += 50;
    } else if (amount > 50000) {
        alerts.push("High-value transaction");
        riskScore += 25;
    }
    
    const locations = ["New York", "London", "Tokyo", "Paris", "Sydney", "Dubai"];
    if (!locations.includes(location)) {
        alerts.push("Unusual location");
        riskScore += 15;
    }
    
    if (amount % 1000 === 0 && amount > 1000) {
        alerts.push("Round amount transaction");
        riskScore += 5;
    }
    
    const status = riskScore > 20 ? "suspicious" : "clean";
    
    if (alerts.length === 0) {
        alerts.push("No fraud detected");
    }
    
    return {
        success: true,
        transaction: {
            id: Date.now(),
            accNo: parseInt(accNo),
            amount: parseFloat(amount),
            location: location,
            timestamp: Math.floor(Date.now() / 1000),
            riskScore: Math.min(riskScore, 100),
            status: status,
            remainingBalance: 100000 - amount,
            phone: mobileNumber,
            email: emailAddress,
            alerts: alerts
        }
    };
}

app.listen(port, () => {
    console.log(`🚀 Fraud Detection System Server running at http://localhost:${port}`);
    console.log('📊 Frontend: http://localhost:3000');
    console.log('🔧 Alert System:');
    console.log('   - SMS Alerts: ≥30% risk score');
    console.log('   - Email Alerts: ≥60% risk score');
    console.log('   - Twilio: ACTIVE with corrected credentials');
    console.log('   - EmailJS: ACTIVE');
    console.log('💡 Test with amount > $50,000 to trigger alerts');
});
