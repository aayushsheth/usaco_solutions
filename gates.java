import java.io.*;
import java.util.*;
public class gates {
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("gates.in"));
		PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("gates.out")));

		int n = Integer.parseInt(br.readLine());
		String s = br.readLine();
		int currX = 1002;
		int currY = 1002;
		boolean[][] isFence = new boolean[2005][2005];
		isFence[currX][currY] = true;
		for(int i = 0; i < s.length(); i++) {
			int dirX = 0, dirY = 0;
			if(s.charAt(i) == 'N') {
				dirX = -1;
			}
			else if(s.charAt(i) == 'S') {
				dirX = 1;
			}
			else if(s.charAt(i) == 'W') {
				dirY = -1;
			}
			else {
				dirY = 1;
			}
			for(int a = 0; a < 2; a++) {
				currX += dirX;
				currY += dirY;
				isFence[currX][currY] = true;
			}
		}
		int ret = -1;
		int[] dx = new int[]{-1,1,0,0};
		int[] dy = new int[]{0,0,-1,1};
		for(int i = 0; i < isFence.length; i++) {
			for(int j = 0; j < isFence[i].length; j++) {
				if(isFence[i][j]) {
					continue;
				}
				ret++;
				LinkedList<Point> q = new LinkedList<Point>();
				q.add(new Point(i, j));
				isFence[i][j] = true;
				while(!q.isEmpty()) {
					Point curr = q.removeFirst();
					for(int k = 0; k < dx.length; k++) {
						int nx = curr.x + dx[k];
						int ny = curr.y + dy[k];
						if(nx >= 0 && nx < isFence.length && ny >= 0 && ny < isFence[nx].length && !isFence[nx][ny]) {
							isFence[nx][ny] = true;
							q.add(new Point(nx, ny));
						}
					}
				}
			}
		}
		pw.println(ret);
		
		pw.close();
	}
	
	static class Point {
		public int x,y;
		public Point(int xIn, int yIn) {
			x = xIn;
			y = yIn;
		}
	}
	
}