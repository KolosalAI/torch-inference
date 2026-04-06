require('dotenv').config({ path: __dirname + '/.env.test' });
const { stopServer } = require('./utils/server');

module.exports = async function globalTeardown() {
  await stopServer();
};
