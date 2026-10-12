const fs = require('fs');
fs.writeFileSync('dump.json', JSON.stringify({argv: process.argv}, null, 2));
