const fs = require('fs');
fs.writeFileSync('args.log', JSON.stringify({argv: process.argv, env: Object.keys(process.env).filter(k => k.startsWith('NL_'))}, null, 2));
