require("dotenv").config();

const procenv = process.env,
  Discord = require("discord.js"),
  client = new Discord.Client({
    intents: [
      "GUILD_MESSAGES",
      "GUILDS",
      "GUILD_EMOJIS_AND_STICKERS",
      "GUILD_MEMBERS",
    ],
  }),
  Enmap = require("enmap"),
  db = new Enmap({
    name: "db",
  }),
  { performance } = require("perf_hooks"),
  py = require("pythonia");

function logger(msg) {
  console.log(`[${new Date()}] ${msg}`);
}

function formatMsToDuration(ms) {
  var seconds = Math.floor(ms / 1000);
  var minutes = Math.floor(seconds / 60);
  var hours = Math.floor(minutes / 60);
  var days = Math.floor(hours / 24);
  var years = Math.floor(days / 365);
  var months = Math.floor(years / 12);
  var weeks = Math.floor(days / 7);
  var days = days % 7;
  var hours = hours % 24;
  var minutes = minutes % 60;
  var seconds = seconds % 60;
  var res = "";
  if (years > 0) res += `${years} year${years > 1 ? "s" : ""} `;
  if (months > 0) res += `${months} month${months > 1 ? "s" : ""} `;
  if (weeks > 0) res += `${weeks} week${weeks > 1 ? "s" : ""} `;
  if (days > 0) res += `${days} day${days > 1 ? "s" : ""} `;
  if (hours > 0) res += `${hours} hour${hours > 1 ? "s" : ""} `;
  if (minutes > 0) res += `${minutes} minute${minutes > 1 ? "s" : ""} `;
  if (seconds > 0) res += `${seconds} second${seconds > 1 ? "s" : ""} `;
  return res;
}

(async () => {
  // Import the query utility
  var start = performance.now();
  const q = await py.python("./utils/query.py");
  logger(`Loaded query.py in ${formatMsToDuration(performance.now() - start)}`);

  async function login() {
    client.login(procenv.TOKEN).catch(() => {
      logger("Login failed, retrying in 5 seconds...");
      setTimeout(login, 5000);
    });
  }

  await login();

  client.on("ready", () => {
    logger(`Logged in as ${client.user.tag}!`);
  });

  client.on("messageCreate", async (message) => {
    if (message.author.bot || !message.guild) return;

    const channelId = message.channel.id;

    // Check if the current channel is in the database and check the number of turns logged
    // If not, add it to the database and set the message as the first turn
    if (!db.has(channelId)) db.set(channelId, [message.content]);

    // Get all the turns stored
    /** @type {string[]} */
    let turns = db.get(channelId);

    // If there are more than 18 turns, remove the oldest turns so it's not over 18
    if (turns.length > 18) turns = turns.slice(turns.length - 18);

    // Concat the turns
    let formattedTurn = turns.join("\n");

    // Generate the query and format so there are no quotes
    const query = (await q.query(formattedTurn)).replace(/"/g, "");

    // Send the query result to the channel
    const resMess = await message.reply({
      content: query,
      allowedMentions: { repliedUser: false },
    });

    // Add the reply to the database
    turns.push(resMess.content ? resMess.content : resMess);

    db.set(channelId, turns);
  });
})();
