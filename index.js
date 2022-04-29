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
  Enmap = require("enmap").default,
  db = new Enmap({
    name: "db",
  }),
  py = require("pythonia");

function login() {
  client.login(procenv.TOKEN).catch(() => {
    console.log("Login failed, retrying in 5 seconds...");
    setTimeout(login, 5000);
  });
}

login();

client.on("ready", () => {
  console.log(`Logged in as ${client.user.tag}!`);
});

client.on("message", async (message) => {
  if (message.author.bot || !message.guild) return;

  // Check if the current channel is in the database and check the number of turns logged
  // If not, add it to the database and set the message as the first turn
  if (!db.has(message.channel.id)) db.set(message.channel.id, [message]);

  // Get all the turns stored
  /** @type {Discord.Message[]} */
  let turns = db.get(message.channel.id);

  // Add the new turn to the database
  turns.push(message);

  // If there are more than 18 turns, remove the two oldest turns
  if (turns.length > 18) turns = turns.slice(1, turns.length - 1);

  db.set(message.channel.id, turns);

  // Format and concat the turns
  let formattedTurn = turns
    .map((turn) => {
      return `${turn.author.username}: ${turn.content}`;
    })
    .join("\n");

  // Import the query utility
  const q = await py.python("./utils/query.py");

  // Generate the query
  const query = await q.query(formattedTurn);

  // Send the query result to the channel
  const resMess = message.channel.send(query);

  // Push the message to the database
  turns.push(resMess);

  // Set the new database
  db.set(message.channel.id, turns);
});
