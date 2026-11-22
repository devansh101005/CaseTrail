// Team page. The work division is written by the team at the end of the project (PLAN.md section 7).
const TEAM = [
  { name: "Devansh Pandey", roll: "2310110461" },
  { name: "Anamika Pal", roll: "2310110037" },
  { name: "Abhinav Bachchas", roll: "2310110383" },
];

export default function Team() {
  return (
    <div className="mx-auto max-w-[1120px] px-4 py-10 md:px-6">
      <p className="mono-label text-line-red-ink">Team · CSD358 Information Retrieval</p>
      <h1 className="display mt-2 text-[40px] md:text-[56px]">Who plotted the trail</h1>
      <div className="mt-10 grid gap-0 border-t-2 border-ink md:grid-cols-3">
        {TEAM.map((member) => (
          <article key={member.roll} className="border-b border-hairline p-6 md:border-r md:last:border-r-0">
            <h2 className="display text-[22px]">{member.name}</h2>
            <p className="mono-label mt-2 text-ink-soft">Roll no. {member.roll}</p>
            <p className="mt-4 text-sm text-ink-soft">Contribution: to be written by the team.</p>
          </article>
        ))}
      </div>
      <p className="mt-8 max-w-[70ch] text-sm">
        Data: IL-TUR benchmark (Joshi et al., ACL 2024), CC BY-NC-SA 4.0. CaseTrail is a research aid, not legal advice.
        The dataset uses the Indian Penal Code, which was replaced by the Bharatiya Nyaya Sanhita from 1 July 2024.
      </p>
    </div>
  );
}
