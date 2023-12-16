import fs from "fs";
// 1 to 1000
for (let i = 1; i <= 1000; i++) {
  const url = `https://nportal.ntut.edu.tw/authImage.do`;
  const fileName = `./data/${i}.png`;
  await fetch(url)
    .then((res) => res.arrayBuffer())
    .then((buffer) => fs.writeFileSync(fileName, Buffer.from(buffer)));
}
